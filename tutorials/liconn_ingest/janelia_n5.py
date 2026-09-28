#!/usr/bin/env python3
"""Janelia ASLM (Navigate / BigDataViewer N5) tile -> uint8 OME-Zarr pyramids.

Pilot for the Kevin Dean lab drops under coding/data/janelia_aslm/. Each
dataset folder holds `CH00_000000.n5` (one BDV setup per tile, uint16, one
z-plane per N5 block), `CH00_000000.xml` and `experiment.yml`.

Per tile this writes two groups with the SAME intensity mapping:

  <name>_sheared.zarr   raw acquisition geometry, (Z, Y, X) as recorded
  <name>_deskew.zarr    plane z shifted by -shear*z px in Y (linear interp)

Intensity, `--intensity-mode`:
  gat (default)  the `scripts/preprocess_liconn.py` default: one shot-noise model
                 (variance = gain*(x-offset)) and one stretch per tile, fitted on
                 evenly spaced raw planes by `connectomics/data/normalization/
                 anscombe.py` (loaded by path, as nd2_source does); Anscombe
                 transform, transformed 1st/99.9th percentiles -> 0/255. No CLAHE.
  linear         legacy: per-plane percentile clip (--clip), no CLAHE.
The fit runs on the RAW (sheared) frames, where the noise model holds. The
deskew is applied to the pre-rounding 0-255 float plane, so it is a pure
geometric resample of `_sheared`.

Geometry (read, not chosen): the XML ViewRegistration "Shearing Transform" is
y' = y + a*z in pixel units (a = -2.380952 = -step/xy_px for this microscope),
and ClearEx's stage-scan path on the same experiment.yml resolves to the same
thing: shear_yz = tan(-45 deg) in physical um with z spacing = stage step
(0.4 um). So the deskewed group's z spacing is the stage step, and the sheared
group's is the XML voxel z (step*sin45, the plane-normal spacing).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zarr_writer  # noqa: E402
from nd2_source import DEFAULT_STATS_PLANES, load_anscombe_module  # noqa: E402
from ingest import _gcs_env, remote_group_complete  # noqa: E402


def read_plane(n5: Path, setup: int, z: int) -> np.ndarray:
    """One (Y, X) uint16 plane: N5 block (0, 0, z), blosc, big-endian."""
    import numcodecs

    b = (n5 / f"setup{setup}/timepoint0/s0/0/0/{z}").read_bytes()
    mode, nd = struct.unpack(">HH", b[:4])
    dims = struct.unpack(">" + "I" * nd, b[4:4 + 4 * nd])
    off = 4 + 4 * nd + (4 if mode == 1 else 0)
    arr = np.frombuffer(numcodecs.Blosc().decode(b[off:]), ">u2").reshape(dims[::-1])
    return arr[0].astype(np.uint16)


def read_geometry(ds: Path, setup: int) -> dict:
    root = ET.parse(ds / "CH00_000000.xml").getroot()
    vs = next(v for v in root.iter("ViewSetup") if v.findtext("id") == str(setup))
    size_xyz = [int(v) for v in vs.findtext("size").split()]
    vox_xyz = [float(v) for v in vs.find("voxelSize").findtext("size").split()]
    reg = next(r for r in root.iter("ViewRegistration") if r.get("setup") == str(setup))
    shear = None
    grid = None
    for t in reg.iter("ViewTransform"):
        m = [float(v) for v in t.findtext("affine").split()]
        if t.get("Name") == "Shearing Transform":
            shear = m[6]                       # row y, column z
        elif t.get("Name") == "Translation to Regular Grid":
            grid = [m[3], m[7], m[11]]
    exp = json.loads((ds / "experiment.yml").read_text())
    return {"size_zyx": size_xyz[::-1], "voxel_um_zyx": vox_xyz[::-1],
            "shear_px_per_plane": shear, "grid_translation_xyz": grid,
            "step_um": float(exp["MicroscopeState"]["step_size"]),
            "label_488nm": exp.get("Saving", {}).get("label_488nm")}


def normalize(plane: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    p = plane.astype(np.float32)
    lo, hi = np.percentile(p, (lo_p, hi_p))
    if hi <= lo:
        return np.zeros(p.shape, np.float32)
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0) * 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    """0-255 float -> uint8 (rint, as anscombe_to_uint8 / preprocess_xy_plane)."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def shift_y(x: np.ndarray, out_ny: int, y0: float) -> np.ndarray:
    """Place `x` so its row 0 lands at output row y0 (fractional), linear interp."""
    ny = x.shape[0]
    i = int(np.floor(y0))
    f = y0 - i
    out = np.zeros((out_ny, x.shape[1]), np.float32)
    out[i:i + ny] += (1.0 - f) * x
    out[i + 1:i + 1 + ny] += f * x[: out_ny - i - 1]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path, help=".../<date>_janelia_.../36x_001")
    p.add_argument("--setup", type=int, default=0)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--intensity-mode", choices=("gat", "linear"), default="gat")
    p.add_argument("--stretch-percentiles", type=float, nargs=2, default=(1.0, 99.9),
                   help="gat: transformed percentiles -> 0/255 (preprocess_liconn default)")
    p.add_argument("--stats-planes", type=int, default=DEFAULT_STATS_PLANES)
    p.add_argument("--noise-model", type=float, nargs=2, default=None,
                   metavar=("GAIN", "OFFSET"), help="gat: skip the fit")
    p.add_argument("--clip", type=float, nargs=2, default=(1.0, 99.5),
                   help="linear: per-plane percentiles")
    p.add_argument("--force", action="store_true", help="rewrite existing local groups")
    p.add_argument("--fold", type=float, default=None,
                   help="expansion fold; default parsed from experiment.yml label_488nm")
    p.add_argument("--bucket", default="donglai_public")
    p.add_argument("--prefix", default="liconn/janelia")
    p.add_argument("--upload", action="store_true")
    p.add_argument("--only", choices=("sheared", "deskew"), default=None)
    a = p.parse_args()

    ds = a.dataset.resolve()
    n5 = ds / "CH00_000000.n5"
    g = read_geometry(ds, a.setup)
    fold = a.fold
    if fold is None:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)x", str(g["label_488nm"] or ""))
        if not m:
            sys.exit(f"no fold in label_488nm={g['label_488nm']!r}; pass --fold")
        fold = float(m.group(1))
    nz, ny, nx = g["size_zyx"]
    s = g["shear_px_per_plane"]
    vz, vy, vx = g["voxel_um_zyx"]
    dataset_id = f"{ds.parent.name}_{ds.name}"
    name = f"{dataset_id}_setup{a.setup:02d}"
    print(json.dumps({**g, "fold": fold, "name": name}, indent=1))

    if a.intensity_mode == "gat":
        ans = load_anscombe_module()

        class _Planes:                      # Z-indexable view for fit_anscombe_mapping
            shape = (nz, ny, nx)

            def __getitem__(self, z):
                return read_plane(n5, a.setup, int(z))

        mapping = ans.fit_anscombe_mapping(
            _Planes(), stats_planes=a.stats_planes,
            stretch_percentiles=tuple(a.stretch_percentiles),
            noise_model=tuple(a.noise_model) if a.noise_model else None)
        low, high = mapping["stretch_range"]
        gain, offset = mapping["gain"], mapping["offset"]
        print(f"noise model ({mapping['noise_model_source']}): var = {gain:.3f}*(x-{offset:.1f}); "
              f"transformed [{low:.2f}, {high:.2f}] -> [0,255]; "
              f"noise sigma = {255.0 / (high - low):.1f} uint8 levels")
        intensity = {"intensity_mode": "gat",
                     "noise_model": "variance = gain * (x - offset)",
                     "noise_gain": gain, "noise_offset": offset,
                     "noise_model_source": mapping["noise_model_source"],
                     "stats_z_indices": [int(v) for v in mapping["stats_z_indices"]],
                     "stretch_percentiles": list(mapping["stretch_percentiles"]),
                     "stretch_range": [low, high],
                     "noise_sigma_uint8_levels": 255.0 / (high - low)}

        def to_float(plane):
            t = ans.anscombe_transform(plane, gain, offset)
            return np.clip((t - low) * (255.0 / (high - low)), 0.0, 255.0)
    else:
        intensity = {"intensity_mode": "linear", "clip_mode": "percentile",
                     "clip_percentiles": [float(v) for v in a.clip], "clip_scope": "per_plane"}

        def to_float(plane):
            return normalize(plane, *a.clip)

    base = {"source": str(n5), "source_setup": a.setup, "source_shape_zyx": [nz, ny, nx],
            "source_dtype": "uint16", **intensity,
            "clahe": False, "expansion_fold": fold,
            "fold_source": "cli" if a.fold else "experiment.yml:Saving.label_488nm",
            "spacing_basis": "biological", "xml_shear_px_per_plane": s,
            "grid_translation_xyz_px": g["grid_translation_xyz"],
            "stage_step_um": g["step_um"]}

    off = abs(s) * (nz - 1)
    out_ny = ny + int(np.ceil(off)) + 1
    variants = {
        "sheared": dict(shape=(nz, ny, nx),
                        spacing=[vz * 1000 / fold, vy * 1000 / fold, vx * 1000 / fold],
                        extra={"geometry": "sheared_raw_acquisition",
                               "spacing_note": "z = XML voxel z (plane-normal spacing); "
                                               "volume is NOT deskewed"}),
        "deskew": dict(shape=(nz, out_ny, nx),
                       spacing=[g["step_um"] * 1000 / fold, vy * 1000 / fold, vx * 1000 / fold],
                       extra={"geometry": "deskewed_yz",
                              "deskew": {"y_out = y + shear*z + offset": True,
                                         "shear_px_per_plane": s, "offset_px": off if s < 0 else 0.0,
                                         "interp": "linear", "fill": 0,
                                         "equivalent": "ClearEx shear_yz_deg=-45, z=stage step"}}),
    }

    for v, spec in variants.items():
        if a.only and v != a.only:
            continue
        dest = a.work / name / f"{name}_{v}.zarr"
        if a.force or not (dest / ".zattrs").exists():
            def planes(v=v):
                for z in range(nz):
                    x = to_float(read_plane(n5, a.setup, z))
                    if v == "sheared":
                        yield to_u8(x)
                    else:
                        y0 = s * z + (off if s < 0 else 0.0)
                        yield to_u8(shift_y(x, out_ny, y0))
            attrs = {**base, **spec["extra"], "axes": "ZYX",
                     "spacing_nm_zyx": spec["spacing"]}
            print(f"== {v}: {spec['shape']} spacing {spec['spacing']} nm")
            zarr_writer.write_pyramid(dest, planes(), spec["shape"], spec["spacing"], attrs)
        else:
            print(f"== {v}: already written {dest}")
        if a.upload:
            url = f"gs://{a.bucket}/{a.prefix}/{dataset_id}/image/{dest.name}"
            subprocess.run(["gcloud", "storage", "rsync", "-r",
                            "--delete-unmatched-destination-objects", str(dest), url],
                           check=True, env=_gcs_env())
            ok, found, expected, why = remote_group_complete(url)
            print(f"{'OK ' if ok else 'BAD'} {url}  {found}/{expected}  {why}")


if __name__ == "__main__":
    main()
