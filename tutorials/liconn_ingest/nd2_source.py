#!/usr/bin/env python3
"""Stream one LICONN ND2 as uint8 ZYX planes.

Two intensity paths. `gat` (the default in ingest.py) fits one noise model and
one stretch per volume from sampled planes (`fit_anscombe`), then maps every
plane with that same pointwise function -- see
`connectomics/data/normalization/anscombe.py`. The legacy `percentile` /
`fixed_intensity` paths clip each plane on its own and apply CLAHE through
`scripts/preprocess_liconn.py::preprocess_xy_plane`.

WHY NOT `connectomics.data.io.read_volume`. Two reasons, both practical.
`connectomics/data/__init__.py` imports the augmentation and dataset packages,
so reaching the ND2 reader drags MONAI and torch into a CPU-only image that
otherwise weighs a few hundred MB. And `read_volume` reads the *whole* array:
a 1199x2304x2304 uint16 ND2 is 12.7 GB before the uint8 copy. The axis
resolution below is a faithful port of `io.py::_nd2_output_axes_and_shape`, and
`test_nd2_source.py` asserts the two agree on the shapes that matter.

THE OPTICS SPACING IS IN THE FILE. The ND2 records the acquisition voxel size;
the expansion fold does not appear anywhere in it (see `naming.py`). So spacing
is read here and fold comes from the filename, and `spacing_source` records
which came from where -- that asymmetry is the whole reason provenance carries
it rather than assuming.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PREPROCESS_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "scripts" / "preprocess_liconn.py",
    Path("/workspace/scripts/preprocess_liconn.py"),
)
_ANSCOMBE_RELATIVE = Path("connectomics") / "data" / "normalization" / "anscombe.py"
_ANSCOMBE_CANDIDATES = (
    Path(__file__).resolve().parents[2] / _ANSCOMBE_RELATIVE,
    Path("/workspace") / _ANSCOMBE_RELATIVE,
)
DEFAULT_STATS_PLANES = 16


def load_anscombe_module():
    """Load the repo's Anscombe normalization by path, not via `connectomics`.

    Importing the package would pull torch and MONAI into this CPU image (see
    the top of this file); `anscombe.py` needs only numpy and scipy, so the
    pixel math stays the repo's rather than a copy.
    """
    for path in _ANSCOMBE_CANDIDATES:
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"anscombe.py not found in {_ANSCOMBE_CANDIDATES}")
    spec = importlib.util.spec_from_file_location("_liconn_anscombe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_preprocess_module():
    """Load `scripts/preprocess_liconn.py` without importing `connectomics`.

    The module's only package-level import is `read_volume`, which this file
    deliberately does not use. A stub satisfies it so the CLAHE math -- the
    part that decides the published pixels -- stays the repo's, not a copy.
    """
    import types

    for path in _PREPROCESS_CANDIDATES:
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"preprocess_liconn.py not found in {_PREPROCESS_CANDIDATES}")

    if "connectomics.data.io" not in sys.modules:
        pkg = types.ModuleType("connectomics")
        pkg.__path__ = []
        data = types.ModuleType("connectomics.data")
        data.__path__ = []
        io = types.ModuleType("connectomics.data.io")

        def _unavailable(*_a, **_k):  # pragma: no cover - guard, never called
            raise RuntimeError(
                "read_volume is stubbed out in the ingest image; ND2 reads go "
                "through nd2_source.iter_planes."
            )

        io.read_volume = _unavailable
        sys.modules.update({"connectomics": pkg, "connectomics.data": data,
                            "connectomics.data.io": io})

    spec = importlib.util.spec_from_file_location("_preprocess_liconn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def output_axes_and_shape(filename: str, sizes: dict) -> tuple[list, tuple]:
    """Port of `io.py::_nd2_output_axes_and_shape`. Same errors, same order."""
    unsupported = [axis for axis in sizes if axis not in {"C", "Z", "Y", "X"}]
    if unsupported:
        details = ", ".join(f"{axis}={sizes[axis]}" for axis in unsupported)
        if "P" in unsupported:
            raise ValueError(
                f"ND2 input must contain one XY position/tile; found {details} in {filename}. "
                "Tile concatenation is not supported."
            )
        raise ValueError(
            f"ND2 input contains unsupported axes ({details}) in {filename}. "
            "Only channel, Z, Y, and X axes are supported."
        )
    if "Y" not in sizes or "X" not in sizes:
        raise ValueError(f"ND2 input must contain Y and X axes: {filename}")
    output_axes = [axis for axis in ("C", "Z", "Y", "X") if axis in sizes]
    return output_axes, tuple(int(sizes[axis]) for axis in output_axes)


def inspect(path: str | Path) -> dict:
    """Shape, dtype and optics spacing, without reading pixels."""
    import nd2

    path = str(path)
    with nd2.ND2File(path) as f:
        sizes = {str(axis): int(size) for axis, size in f.sizes.items()}
        axes, shape = output_axes_and_shape(path, sizes)
        # nd2 reports voxel size XYZ in micrometres.
        vx, vy, vz = (float(v) for v in f.voxel_size())
        info = {
            "axes": "".join(axes),
            "shape": list(shape),
            "dtype": str(f.dtype),
            "optics_spacing_zyx_nm": [vz * 1000.0, vy * 1000.0, vx * 1000.0],
            "n_channels": int(sizes.get("C", 1)),
        }
    info["shape_zyx"] = info["shape"][1:] if info["axes"].startswith("C") else info["shape"]
    return info


def channel_exposures_ms(path: str | Path) -> dict:
    """`{channel_index: {"name": str, "exposure_ms": float}}` from ND2 metadata.

    NOT from the filename -- spec.md I7. `ExPID71_120ms-30ms_600nm_40XW02.nd2`
    lists 120 first, but channel 0 is the 30 ms acquisition; the filename order
    is the opposite of the channel order, which is precisely the trap I7 exists
    for.

    The nd2 Python API exposes no structured per-channel exposure for these
    files (checked 2026-09-21: neither `metadata.channels[i].channel` nor
    `frame_metadata(0).channels[i]` carries one), so the per-plane block of
    `text_info['description']` is the only record there is. Parsing free text
    is fragile, so the result is cross-checked against the structured channel
    NAMES, which the API does expose: if the description's plane order does not
    line up with the real channel order, this raises instead of silently
    attaching the wrong exposure to a published volume.
    """
    import re

    import nd2

    with nd2.ND2File(str(path)) as f:
        desc = (f.text_info or {}).get("description", "")
        names = [c.channel.name for c in (f.metadata.channels or [])]

    parsed: dict[int, dict] = {}
    blocks = re.split(r"Plane\s+#(\d+):", desc)
    for i in range(1, len(blocks), 2):
        body = blocks[i + 1]
        name = re.search(r"Name:\s*(\S+)", body)
        exposure = re.search(r"Exposure:\s*([\d.]+)\s*ms", body)
        if not (name and exposure):
            continue
        parsed[int(blocks[i]) - 1] = {"name": name.group(1),
                                      "exposure_ms": float(exposure.group(1))}

    if not parsed:
        raise ValueError(f"no per-plane exposure found in the ND2 metadata of {path}")
    got = [parsed[k]["name"] for k in sorted(parsed)]
    if got != names:
        raise ValueError(
            f"ND2 description plane order {got} does not match the channel order "
            f"{names} in {path}; refusing to guess which exposure is which channel")
    return parsed


def _structural_lazy(f, path: str, channel: int):
    """The open ND2's structural channel as a lazy ZYX dask array."""
    sizes = {str(axis): int(size) for axis, size in f.sizes.items()}
    axes, _shape = output_axes_and_shape(path, sizes)
    n_c = sizes.get("C", 1)
    if channel >= n_c:
        raise ValueError(f"channel {channel} out of range for {n_c} channels in {path}")
    lazy = f.to_dask()                      # input axis order, lazy
    input_axes = list(sizes)
    permutation = tuple(input_axes.index(a) for a in axes)
    lazy = lazy.transpose(permutation)      # -> (C,)Z,Y,X
    if axes[0] == "C":
        lazy = lazy[channel]
    elif channel != 0:
        raise ValueError(f"{path} has no channel axis; use --channel 0")
    return lazy


def fit_anscombe(path: str | Path, channel: int = 0, *, stretch_percentiles=(1.0, 99.9),
                 stats_planes: int = DEFAULT_STATS_PLANES) -> dict:
    """Fit the volume's noise model and stretch; returns a JSON-ready mapping.

    Reads only `stats_planes` evenly spaced planes. The result is what
    `iter_planes(clip_mode="gat", mapping=...)` applies to every plane and what
    `zarr_writer.provenance_attrs` records.
    """
    import nd2

    ans = load_anscombe_module()
    path = str(path)
    with nd2.ND2File(path) as f:
        mapping = ans.fit_anscombe_mapping(
            _structural_lazy(f, path, channel), stats_planes=stats_planes,
            stretch_percentiles=tuple(float(v) for v in stretch_percentiles))
    return anscombe_mapping_json(mapping)


def anscombe_mapping_json(mapping: dict) -> dict:
    """`fit_anscombe_mapping` output as plain JSON types, for .zattrs/provenance."""
    return {"gain": float(mapping["gain"]), "offset": float(mapping["offset"]),
            "noise_model_source": mapping["noise_model_source"],
            "stretch_percentiles": [float(v) for v in mapping["stretch_percentiles"]],
            "stretch_range": [float(v) for v in mapping["stretch_range"]],
            "stats_z_indices": [int(v) for v in mapping["stats_z_indices"]]}


def plane_mapper(clip_mode: str, clip_values=(1.0, 99.0), clip_limit: float | None = 0.03,
                 mapping: dict | None = None):
    """The raw-plane -> uint8-plane function for one volume.

    `gat` applies the volume's fixed Anscombe mapping (fitted up front, so
    streaming stays exactly equivalent to a whole-volume pass). The legacy modes
    compute their percentiles per plane, which is streaming-safe for the same
    reason: no plane's output depends on another plane.
    """
    if clip_mode == "gat":
        if mapping is None:
            raise ValueError("clip_mode 'gat' needs the volume mapping from fit_anscombe")
        if clip_limit is not None:
            raise ValueError("CLAHE does not apply to clip_mode 'gat'; unset clip_limit")
        ans = load_anscombe_module()
        gain, offset = mapping["gain"], mapping["offset"]
        stretch = tuple(mapping["stretch_range"])
        return lambda plane: ans.anscombe_to_uint8(plane, gain=gain, offset=offset,
                                                   stretch_range=stretch)

    pre = load_preprocess_module()
    kwargs = {"clip_limit": clip_limit}
    if clip_mode == "percentile":
        kwargs.update(clip_intensity_range=None,
                      lower_percentile=float(clip_values[0]),
                      upper_percentile=float(clip_values[1]))
    elif clip_mode == "fixed_intensity":
        kwargs.update(clip_intensity_range=(float(clip_values[0]), float(clip_values[1])))
    else:
        raise ValueError(f"unknown clip_mode {clip_mode!r}")
    return lambda plane: pre.preprocess_xy_plane(plane, **kwargs)


def iter_planes(path: str | Path, channel: int = 0, *, clip_mode: str = "percentile",
                clip_values=(1.0, 99.0), clip_limit: float | None = 0.03,
                mapping: dict | None = None):
    """Yield uint8 YX planes in Z order; see `plane_mapper` for the intensity path."""
    import nd2

    to_uint8 = plane_mapper(clip_mode, clip_values, clip_limit, mapping)
    path = str(path)
    with nd2.ND2File(path) as f:
        lazy = _structural_lazy(f, path, channel)
        for z in range(lazy.shape[0]):
            yield to_uint8(np.asarray(lazy[z]))
