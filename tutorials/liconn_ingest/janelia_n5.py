#!/usr/bin/env python3
"""Janelia ASLM (Navigate / BigDataViewer N5) tile -> uint8 OME-Zarr pyramids.

Pilot for the Kevin Dean lab drops under coding/data/janelia_aslm/. Each
dataset folder holds `CH00_000000.n5` (one BDV setup per tile, uint16, one
z-plane per N5 block), `CH00_000000.xml` and `experiment.yml`.

Per tile this writes two groups with the SAME intensity mapping:

  <name>_sheared.zarr   raw acquisition geometry, (Z, Y, X) as recorded
  <name>_deskew.zarr    projected stage-scan YZ shear, resampled along Y

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

For this stage-Z scan, project the mechanical step onto camera Y and Z using
the recorded scan angle. The XML Y shear uses the unprojected step, whereas
its Z spacing is already projected; do not mix these two conventions. Full
coverage retains a diagonal valid-data footprint in the output bounding box.
"""

from __future__ import annotations

import argparse
import json
import math
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
    state = exp["MicroscopeState"]
    bdv_shear = exp["BDVParameters"]["shear"]
    if not bdv_shear["shear_data"] or bdv_shear["shear_dimension"] != "YZ":
        raise ValueError("expected a Navigate YZ stage-scan dataset")
    return {"size_zyx": size_xyz[::-1], "voxel_um_zyx": vox_xyz[::-1],
            "shear_px_per_plane": shear, "grid_translation_xyz": grid,
            "step_um": float(state["step_size"]),
            "scan_axis": state["primary_z_axis"],
            "image_mode": state["image_mode"],
            "waveform_template": state["waveform_template"],
            "shear_angle_deg": float(bdv_shear["shear_angle"]),
            "label_488nm": exp.get("Saving", {}).get("label_488nm")}


def resolve_deskew_calibration(g: dict, shear_override: float | None = None) -> dict:
    """Project a mechanical stage-Z step, as in PetaKit's stage-scan deskew.

    This is deliberately limited to the ordinary Navigate z-stack acquisition.
    Constant-velocity scans can encode an optical, already-projected step.
    """
    if (g["scan_axis"] != "z" or g["image_mode"] != "z-stack"
            or g["waveform_template"] != "Default"):
        raise ValueError("projected deskew requires a Default z-stack on stage z")
    step = g["step_um"]
    vz, vy, vx = g["voxel_um_zyx"]
    if any(not math.isfinite(v) or v <= 0 for v in (step, vz, vy, vx)):
        raise ValueError("stage step and voxel spacings must be finite and positive")
    angle = (g["shear_angle_deg"] + 180.0) % 360.0 - 180.0
    if not math.isfinite(angle) or not 0 < abs(angle) <= 90:
        raise ValueError("stage-scan angle must have magnitude in (0, 90] degrees")
    theta = math.radians(abs(angle))
    z_spacing = step * math.sin(theta)
    if not math.isclose(vz, z_spacing, rel_tol=1e-5, abs_tol=1e-9):
        raise ValueError("XML Z spacing disagrees with the projected stage step")
    lateral_step = 0.0 if abs(angle) == 90.0 else step * math.cos(theta)
    shear = math.copysign(lateral_step / vy, angle)
    if shear_override is not None:
        if not math.isfinite(shear_override):
            raise ValueError("shear override must be finite")
        shear = shear_override
    return {"shear_px_per_plane": shear, "z_spacing_um": z_spacing,
            "angle_deg": angle,
            "shear_source": "stage_projection" if shear_override is None else "cli"}


def normalize(plane: np.ndarray, lo_p: float, hi_p: float) -> np.ndarray:
    p = plane.astype(np.float32)
    lo, hi = np.percentile(p, (lo_p, hi_p))
    if hi <= lo:
        return np.zeros(p.shape, np.float32)
    return np.clip((p - lo) / (hi - lo), 0.0, 1.0) * 255.0


def to_u8(x: np.ndarray) -> np.ndarray:
    """0-255 float -> uint8 (rint, as anscombe_to_uint8 / preprocess_xy_plane)."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def deskew_geometry(nz: int, ny: int, shear: float) -> dict:
    """Return the pixel bounds of a full-volume YZ shear."""
    if not math.isfinite(shear):
        raise ValueError("shear must be finite")
    if nz < 1 or ny < 1:
        raise ValueError("source dimensions must be positive")
    extent = abs(shear) * (nz - 1)
    return {"shape_zy": (nz, ny + math.ceil(extent)),
            "y_offset": max(0.0, -shear * (nz - 1)),
            "shear_px_per_plane": shear}


def deskew_planes(get_plane, shape_zyx: tuple[int, int, int], geometry: dict):
    """Inverse-map Y and normalize partial boundary support as ClearEx does."""
    nz, ny, nx = shape_zyx
    out_ny = geometry["shape_zy"][1]
    output_rows = np.arange(out_ny)
    for z in range(nz):
        source_rows = output_rows - geometry["y_offset"] - geometry["shear_px_per_plane"] * z
        lower = np.floor(source_rows).astype(np.int64)
        fraction = source_rows - lower
        upper = lower + 1
        valid_lower = (lower >= 0) & (lower < ny)
        valid_upper = (upper >= 0) & (upper < ny)
        plane = get_plane(z)
        out = np.zeros((out_ny, nx), np.float32)
        support = np.zeros(out_ny, np.float32)
        out[valid_lower] += ((1.0 - fraction[valid_lower, None])
                             * plane[lower[valid_lower]])
        support[valid_lower] += 1.0 - fraction[valid_lower]
        out[valid_upper] += fraction[valid_upper, None] * plane[upper[valid_upper]]
        support[valid_upper] += fraction[valid_upper]
        np.divide(out, support[:, None], out=out, where=support[:, None] > 1e-6)
        yield to_u8(out)


def check_existing_deskew(dest: Path, shear: float, z_spacing_um: float) -> None:
    """Do not silently reuse a differently transformed group."""
    attrs = json.loads((dest / ".zattrs").read_text())
    deskew = attrs.get("deskew", {})
    existing = deskew.get("shear_px_per_plane")
    existing_z = deskew.get("z_spacing_um")
    if (deskew.get("method") != "projected_stage_scan_y_shear" or existing is None
            or existing_z is None
            or not math.isclose(float(existing), shear, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(float(existing_z), z_spacing_um, rel_tol=0, abs_tol=1e-9)):
        raise ValueError(
            f"{dest} has an older or different deskew transform; pass --force to rewrite"
        )


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
    p.add_argument("--shear-px-per-plane", type=float, default=None,
                   help="override projected Y shift per Z plane; keeps calibrated Z spacing")
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
    xml_shear = g["shear_px_per_plane"]
    calibration = resolve_deskew_calibration(g, a.shear_px_per_plane)
    s = calibration["shear_px_per_plane"]
    vz, vy, vx = g["voxel_um_zyx"]
    geometry = deskew_geometry(nz, ny, s)
    out_nz, out_ny = geometry["shape_zy"]
    dataset_id = f"{ds.parent.name}_{ds.name}"
    name = f"{dataset_id}_setup{a.setup:02d}"
    print(json.dumps({**g, "fold": fold, "name": name, "deskew": calibration}, indent=1))

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
        gain, noise_offset = mapping["gain"], mapping["offset"]
        print(f"noise model ({mapping['noise_model_source']}): "
              f"var = {gain:.3f}*(x-{noise_offset:.1f}); "
              f"transformed [{low:.2f}, {high:.2f}] -> [0,255]; "
              f"noise sigma = {255.0 / (high - low):.1f} uint8 levels")
        intensity = {"intensity_mode": "gat",
                     "noise_model": "variance = gain * (x - offset)",
                     "noise_gain": gain, "noise_offset": noise_offset,
                     "noise_model_source": mapping["noise_model_source"],
                     "stats_z_indices": [int(v) for v in mapping["stats_z_indices"]],
                     "stretch_percentiles": list(mapping["stretch_percentiles"]),
                     "stretch_range": [low, high],
                     "noise_sigma_uint8_levels": 255.0 / (high - low)}

        def to_float(plane):
            t = ans.anscombe_transform(plane, gain, noise_offset)
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
            "spacing_basis": "biological", "xml_shear_px_per_plane": xml_shear,
            "grid_translation_xyz_px": g["grid_translation_xyz"],
            "stage_step_um": g["step_um"]}

    deskew_attrs = {"method": "projected_stage_scan_y_shear",
                    "y_out = y + shear*z + offset": True,
                    "offset_px": geometry["y_offset"],
                    "rotation_deg": 0.0,
                    **calibration,
                    "source_to_output_zyx": [[1, 0, 0, 0],
                                             [s, 1, 0, geometry["y_offset"]],
                                             [0, 0, 1, 0], [0, 0, 0, 1]],
                    "interp": "linear_y_support_normalized", "fill": 0}

    variants = {
        "sheared": dict(shape=(nz, ny, nx),
                        spacing=[vz * 1000 / fold, vy * 1000 / fold, vx * 1000 / fold],
                        extra={"geometry": "sheared_raw_acquisition",
                               "spacing_note": "z = XML voxel z (plane-normal spacing); "
                                               "volume is NOT deskewed"}),
        "deskew": dict(shape=(out_nz, out_ny, nx),
                       spacing=[calibration["z_spacing_um"] * 1000 / fold, vy * 1000 / fold,
                                vx * 1000 / fold],
                       extra={"geometry": "deskewed_yz",
                              "deskew": deskew_attrs}),
    }

    for v, spec in variants.items():
        if a.only and v != a.only:
            continue
        dest = a.work / name / f"{name}_{v}.zarr"
        if a.force or not (dest / ".zattrs").exists():
            if v == "sheared":
                planes = (to_u8(to_float(read_plane(n5, a.setup, z)))
                          for z in range(nz))
            else:
                planes = deskew_planes(
                    lambda z: to_float(read_plane(n5, a.setup, z)),
                    (nz, ny, nx), geometry)
            attrs = {**base, **spec["extra"], "axes": "ZYX",
                     "spacing_nm_zyx": spec["spacing"]}
            print(f"== {v}: {spec['shape']} spacing {spec['spacing']} nm")
            zarr_writer.write_pyramid(dest, planes, spec["shape"], spec["spacing"], attrs)
        else:
            if v == "deskew":
                check_existing_deskew(dest, s, calibration["z_spacing_um"])
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
