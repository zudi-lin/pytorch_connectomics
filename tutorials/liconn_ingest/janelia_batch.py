#!/usr/bin/env python3
"""Plan, run, and verify one Janelia N5 setup per batch task."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import janelia_n5


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def plan(raw, work, manifest):
    raw, work = Path(raw).resolve(), Path(work).resolve()
    tasks, names = [], set()
    for xml in sorted(raw.glob("*/*/CH00_000000.xml")):
        dataset = xml.parent
        for view in ET.parse(xml).getroot().iter("ViewSetup"):
            setup = int(view.findtext("id"))
            geometry = janelia_n5.read_geometry(dataset, setup)
            calibration = janelia_n5.resolve_deskew_calibration(geometry)
            fold_match = re.fullmatch(r"(\d+(?:\.\d+)?)x", str(geometry["label_488nm"] or ""))
            if fold_match is None or float(fold_match[1]) <= 0:
                raise ValueError(f"{dataset}: invalid expansion label")
            fold = float(fold_match[1])
            nz, ny, nx = geometry["size_zyx"]
            level = dataset / f"CH00_000000.n5/setup{setup}/timepoint0/s0"
            attrs = read_json(level / "attributes.json")
            if (
                attrs["dimensions"] != [nx, ny, nz]
                or attrs["blockSize"] != [nx, ny, 1]
                or attrs["dataType"] != "uint16"
                or attrs["compression"]["type"] != "blosc"
            ):
                raise ValueError(f"{level}: unsupported N5 block layout")
            missing = [
                z
                for z in range(nz)
                if not (level / f"0/0/{z}").is_file() or (level / f"0/0/{z}").stat().st_size == 0
            ]
            if missing:
                raise ValueError(f"{level}: {len(missing)} missing/empty planes: {missing[:10]}")
            name = f"{dataset.parent.name}_{dataset.name}_setup{setup:02d}"
            if name in names:
                raise ValueError(f"duplicate output name: {name}")
            names.add(name)
            shear = calibration["shear_px_per_plane"]
            deskew_shape = janelia_n5.deskew_geometry(nz, ny, shear)["shape_zy"]
            spacing = [v * 1000 / fold for v in geometry["voxel_um_zyx"]]
            tasks.append(
                {
                    "dataset": str(dataset),
                    "setup": setup,
                    "name": name,
                    "shape_zyx": [nz, ny, nx],
                    "spacing_nm_zyx": spacing,
                    "deskew_shape_zyx": [*deskew_shape, nx],
                    "deskew_spacing_nm_zyx": [
                        calibration["z_spacing_um"] * 1000 / fold,
                        *spacing[1:],
                    ],
                    "shear_px_per_plane": shear,
                    "z_spacing_um": calibration["z_spacing_um"],
                }
            )
    if not tasks:
        raise ValueError(f"no Janelia XML setups found under {raw}")
    write_json(manifest, {"raw_root": str(raw), "work": str(work), "tasks": tasks})
    print(f"Planned {len(tasks)} setups in {manifest}", flush=True)


def run(manifest, index):
    batch = read_json(manifest)
    if not 0 <= index < len(batch["tasks"]):
        raise ValueError(f"task index out of range: {index}")
    task = batch["tasks"][index]
    status_path = Path(manifest).parent / "status" / f"{index}.json"
    status = {
        "index": index,
        "name": task["name"],
        "state": "started",
        "started_at": timestamp(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    }
    write_json(status_path, status)
    argv = sys.argv
    try:
        from numcodecs import blosc

        blosc.set_nthreads(1)
        sys.argv = [
            "janelia_n5.py",
            task["dataset"],
            "--setup",
            str(task["setup"]),
            "--work",
            batch["work"],
        ]
        janelia_n5.main()
    except BaseException as error:
        status.update(
            state="failed", finished_at=timestamp(), error=f"{type(error).__name__}: {error}"
        )
        write_json(status_path, status)
        raise
    else:
        status.update(state="completed", finished_at=timestamp())
        write_json(status_path, status)
    finally:
        sys.argv = argv


def verify(manifest):
    batch = read_json(manifest)
    issues, counts = [], Counter()
    for index, task in enumerate(batch["tasks"]):
        try:
            status = read_json(Path(manifest).parent / "status" / f"{index}.json")
            counts[status["state"]] += 1
            if status["state"] != "completed" or status["name"] != task["name"]:
                raise ValueError(f"task status is {status}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"index": index, "variant": "status", "error": str(error)})
        for variant in ("sheared", "deskew"):
            dest = Path(batch["work"]) / task["name"] / f"{task['name']}_{variant}.zarr"
            try:
                attrs = read_json(dest / ".zattrs")
                level = read_json(dest / "0/.zarray")
                prefix = "deskew_" if variant == "deskew" else ""
                if level["shape"] != task[f"{prefix}shape_zyx"] or level["dtype"] != "|u1":
                    raise ValueError("level 0 shape or dtype mismatch")
                expected = task[f"{prefix}spacing_nm_zyx"]
                transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
                scale = next(t["scale"] for t in transforms if t["type"] == "scale")
                for actual in (attrs["spacing_nm_zyx"], scale):
                    if len(actual) != 3 or not all(
                        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                        for a, b in zip(actual, expected)
                    ):
                        raise ValueError("voxel spacing mismatch")
                if variant == "deskew":
                    janelia_n5.check_existing_deskew(
                        dest, task["shear_px_per_plane"], task["z_spacing_um"]
                    )
            except (OSError, ValueError, KeyError, IndexError, StopIteration, TypeError) as error:
                issues.append({"index": index, "variant": variant, "error": str(error)})
    summary = {
        "verified_at": timestamp(),
        "total": len(batch["tasks"]),
        "status_counts": dict(counts),
        "ok": not issues,
        "issues": issues,
    }
    write_json(Path(manifest).parent / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if issues else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--raw", type=Path, required=True)
    planner.add_argument("--work", type=Path, required=True)
    for command in (planner, commands.add_parser("run"), commands.add_parser("verify")):
        command.add_argument("--manifest", type=Path, required=True)
        if command.prog.endswith(" run"):
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.raw, args.work, args.manifest)
    elif args.command == "run":
        run(args.manifest, args.index)
    else:
        return verify(args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
