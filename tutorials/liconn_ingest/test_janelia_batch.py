"""Batch scheduling contracts without Slurm or real image conversion."""

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import janelia_batch as batch  # noqa: E402


def _fixture(root, duplicate=False):
    dataset = root / "raw/acquisition/tile"
    level = dataset / "CH00_000000.n5/setup0/timepoint0/s0"
    batch.write_json(
        level / "attributes.json",
        {
            "dimensions": [4, 3, 2],
            "blockSize": [4, 3, 1],
            "dataType": "uint16",
            "compression": {"type": "blosc"},
        },
    )
    (level / "0/0").mkdir(parents=True)
    for z in range(2):
        (level / f"0/0/{z}").write_bytes(b"placeholder")
    view = """<ViewSetup><id>0</id><size>4 3 2</size><voxelSize>
        <size>0.168 0.168 0.28284271247461906</size></voxelSize></ViewSetup>"""
    (dataset / "CH00_000000.xml").write_text(
        "<SpimData>"
        + view * (2 if duplicate else 1)
        + '<ViewRegistration setup="0"><ViewTransform Name="Shearing Transform">'
        + "<affine>1 0 0 0 0 1 -2.380952 0 0 0 1 0</affine>"
        + "</ViewTransform></ViewRegistration></SpimData>"
    )
    batch.write_json(
        dataset / "experiment.yml",
        {
            "Saving": {"label_488nm": "22x"},
            "MicroscopeState": {
                "step_size": 0.4,
                "primary_z_axis": "z",
                "image_mode": "z-stack",
                "waveform_template": "Default",
            },
            "BDVParameters": {
                "shear": {"shear_data": True, "shear_dimension": "YZ", "shear_angle": 315}
            },
        },
    )
    return root / "raw", root / "work", root / "batch/manifest.json", level


def _outputs(manifest):
    plan = batch.read_json(manifest)
    task = plan["tasks"][0]
    batch.write_json(
        manifest.parent / "status/0.json", {"state": "completed", "name": task["name"]}
    )
    for variant in ("sheared", "deskew"):
        prefix = "deskew_" if variant == "deskew" else ""
        dest = Path(plan["work"]) / task["name"] / f"{task['name']}_{variant}.zarr"
        spacing = task[f"{prefix}spacing_nm_zyx"]
        batch.write_json(dest / "0/.zarray", {"shape": task[f"{prefix}shape_zyx"], "dtype": "|u1"})
        batch.write_json(
            dest / ".zattrs",
            {
                "spacing_nm_zyx": spacing,
                "multiscales": [
                    {
                        "datasets": [
                            {"coordinateTransformations": [{"type": "scale", "scale": spacing}]}
                        ]
                    }
                ],
                "deskew": {
                    "method": "projected_stage_scan_y_shear",
                    "shear_px_per_plane": task["shear_px_per_plane"],
                    "z_spacing_um": task["z_spacing_um"],
                },
            },
        )


def test_plan_uses_xml_setups_not_extra_n5_metadata():
    with tempfile.TemporaryDirectory() as directory:
        raw, work, manifest, level = _fixture(Path(directory))
        (level.parents[2] / "setup99").mkdir()
        batch.plan(raw, work, manifest)
        tasks = batch.read_json(manifest)["tasks"]
        assert len(tasks) == 1 and tasks[0]["setup"] == 0
        assert tasks[0]["shape_zyx"] == [2, 3, 4]
        assert tasks[0]["deskew_shape_zyx"] == [2, 5, 4]


def test_plan_rejects_missing_source_plane():
    with tempfile.TemporaryDirectory() as directory:
        raw, work, manifest, level = _fixture(Path(directory))
        (level / "0/0/1").unlink()
        with TestCase().assertRaisesRegex(ValueError, "missing/empty planes"):
            batch.plan(raw, work, manifest)
        assert not manifest.exists()


def test_plan_rejects_duplicate_output():
    with tempfile.TemporaryDirectory() as directory:
        raw, work, manifest, _ = _fixture(Path(directory), duplicate=True)
        with TestCase().assertRaisesRegex(ValueError, "duplicate output"):
            batch.plan(raw, work, manifest)


def test_run_records_completion_and_failure_and_restores_argv():
    with tempfile.TemporaryDirectory() as directory:
        raw, work, manifest, _ = _fixture(Path(directory))
        batch.plan(raw, work, manifest)
        original = sys.argv
        with patch.object(batch.janelia_n5, "main") as main:
            batch.run(manifest, 0)
            main.assert_called_once_with()
        status = batch.read_json(manifest.parent / "status/0.json")
        assert status["state"] == "completed" and status["finished_at"]
        assert sys.argv is original
        with patch.object(batch.janelia_n5, "main", side_effect=RuntimeError("broken")):
            with TestCase().assertRaisesRegex(RuntimeError, "broken"):
                batch.run(manifest, 0)
        status = batch.read_json(manifest.parent / "status/0.json")
        assert status["state"] == "failed" and "broken" in status["error"]
        assert sys.argv is original


def test_verify_reports_missing_and_invalid_outputs():
    with tempfile.TemporaryDirectory() as directory:
        raw, work, manifest, _ = _fixture(Path(directory))
        batch.plan(raw, work, manifest)
        assert batch.verify(manifest) == 1
        summary = batch.read_json(manifest.parent / "summary.json")
        assert {issue["variant"] for issue in summary["issues"]} == {"status", "sheared", "deskew"}
        _outputs(manifest)
        assert batch.verify(manifest) == 0
        task = batch.read_json(manifest)["tasks"][0]
        metadata = work / task["name"] / f"{task['name']}_deskew.zarr/0/.zarray"
        values = json.loads(metadata.read_text())
        values["shape"] = [99, 99, 99]
        batch.write_json(metadata, values)
        assert batch.verify(manifest) == 1
        summary = batch.read_json(manifest.parent / "summary.json")
        assert len(summary["issues"]) == 1
        assert summary["issues"][0]["variant"] == "deskew"
