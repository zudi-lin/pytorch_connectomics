"""Small contract tests for the Janelia YZ deskew."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import janelia_n5 as janelia  # noqa: E402


def _stage_scan_metadata(angle_deg=-45.0):
    return {
        "voxel_um_zyx": [0.4 * np.sin(np.deg2rad(abs(angle_deg))), 0.168, 0.168],
        "step_um": 0.4,
        "shear_px_per_plane": 0.4 * np.tan(np.deg2rad(angle_deg)) / 0.168,
        "shear_angle_deg": angle_deg % 360,
        "scan_axis": "z",
        "image_mode": "z-stack",
        "waveform_template": "Default",
    }


def test_read_geometry_resolves_recorded_315_degree_stage_scan():
    xml = """<SpimData>
        <SequenceDescription><ViewSetups><ViewSetup>
        <id>0</id><size>2048 648 500</size>
        <voxelSize><unit>um</unit><size>0.168 0.168 0.28284271247461906</size></voxelSize>
        </ViewSetup></ViewSetups></SequenceDescription>
        <ViewRegistrations><ViewRegistration timepoint="0" setup="0">
        <ViewTransform Name="Translation to Regular Grid">
        <affine>1 0 0 100 0 1 0 200 0 0 1 300</affine>
        </ViewTransform><ViewTransform Name="Shearing Transform">
        <affine>1 0 0 0 0 1 -2.380952 0 0 0 1 0</affine>
        </ViewTransform></ViewRegistration></ViewRegistrations>
        </SpimData>"""
    experiment = {
        "Saving": {"label_488nm": "22x"},
        "MicroscopeState": {
            "step_size": 0.4,
            "primary_z_axis": "z",
            "image_mode": "z-stack",
            "waveform_template": "Default",
        },
        "BDVParameters": {
            "shear": {"shear_data": True, "shear_dimension": "YZ", "shear_angle": 315.0}
        },
    }
    with tempfile.TemporaryDirectory() as work:
        tmp_path = Path(work)
        (tmp_path / "CH00_000000.xml").write_text(xml)
        (tmp_path / "experiment.yml").write_text(json.dumps(experiment))
        metadata = janelia.read_geometry(tmp_path, 0)
    assert metadata["size_zyx"] == [500, 648, 2048]
    assert metadata["voxel_um_zyx"] == [0.28284271247461906, 0.168, 0.168]
    assert metadata["grid_translation_xyz"] == [100, 200, 300]
    assert metadata["shear_angle_deg"] == 315.0
    assert metadata["shear_px_per_plane"] == -2.380952
    calibration = janelia.resolve_deskew_calibration(metadata)
    assert calibration["angle_deg"] == -45.0
    np.testing.assert_allclose(calibration["shear_px_per_plane"], -1.683587574)


def test_stage_scan_projects_mechanical_step_into_camera_coordinates():
    for angle_deg in [-60.0, -45.0, -30.0, 30.0, 45.0, 60.0]:
        calibration = janelia.resolve_deskew_calibration(_stage_scan_metadata(angle_deg))
        lateral_um = calibration["shear_px_per_plane"] * 0.168
        axial_um = calibration["z_spacing_um"]
        np.testing.assert_allclose(np.hypot(lateral_um, axial_um), 0.4)
        assert np.sign(lateral_um) == np.sign(angle_deg)
        np.testing.assert_allclose(axial_um, 0.4 * np.sin(np.deg2rad(abs(angle_deg))))
        np.testing.assert_allclose(calibration["angle_deg"], angle_deg)


def test_perpendicular_scan_does_not_add_an_empty_row():
    for angle_deg in [-90.0, 90.0]:
        calibration = janelia.resolve_deskew_calibration(_stage_scan_metadata(angle_deg))
        assert calibration["shear_px_per_plane"] == 0.0
        geometry = janelia.deskew_geometry(5, 4, calibration["shear_px_per_plane"])
        assert geometry["shape_zy"] == (5, 4)


def test_janelia_45_degree_scan_uses_projected_step():
    calibration = janelia.resolve_deskew_calibration(_stage_scan_metadata())
    np.testing.assert_allclose(calibration["shear_px_per_plane"], -1.683587574)
    np.testing.assert_allclose(calibration["z_spacing_um"], 0.282842712)


def test_shear_override_preserves_projected_axial_spacing():
    calibration = janelia.resolve_deskew_calibration(_stage_scan_metadata(), shear_override=-1.9)
    assert calibration["shear_px_per_plane"] == -1.9
    np.testing.assert_allclose(calibration["z_spacing_um"], 0.282842712)
    assert calibration["shear_source"] == "cli"


def test_stage_scan_rejects_unvalidated_acquisition_modes():
    for key, value in [
        ("scan_axis", "x"),
        ("scan_axis", "y"),
        ("image_mode", "single"),
        ("waveform_template", "CVACONPRO"),
    ]:
        metadata = _stage_scan_metadata()
        metadata[key] = value
        with np.testing.assert_raises_regex(ValueError, "Default z-stack on stage z"):
            janelia.resolve_deskew_calibration(metadata)


def test_stage_scan_rejects_invalid_scan_angle():
    for angle in [0.0, 91.0, 180.0, 269.0, np.nan, np.inf]:
        metadata = _stage_scan_metadata()
        metadata["shear_angle_deg"] = angle
        with np.testing.assert_raises(ValueError):
            janelia.resolve_deskew_calibration(metadata)


def test_stage_scan_rejects_nonfinite_shear_override():
    for value in [np.nan, np.inf, -np.inf]:
        with np.testing.assert_raises(ValueError):
            janelia.resolve_deskew_calibration(_stage_scan_metadata(), shear_override=value)


def test_stage_scan_rejects_invalid_voxel_spacing():
    for axis in [0, 1, 2]:
        for value in [0.0, -0.1, np.nan, np.inf]:
            metadata = _stage_scan_metadata()
            metadata["voxel_um_zyx"][axis] = value
            with np.testing.assert_raises(ValueError):
                janelia.resolve_deskew_calibration(metadata)


def test_stage_scan_rejects_invalid_mechanical_step():
    for value in [0.0, -0.4, np.nan, np.inf]:
        metadata = _stage_scan_metadata()
        metadata["step_um"] = value
        with np.testing.assert_raises(ValueError):
            janelia.resolve_deskew_calibration(metadata)


def test_stage_scan_rejects_axial_spacing_inconsistent_with_angle():
    metadata = _stage_scan_metadata()
    metadata["voxel_um_zyx"][0] = 0.4
    with np.testing.assert_raises(ValueError):
        janelia.resolve_deskew_calibration(metadata)


def test_deskew_recovers_spherical_object_from_oblique_acquisition():
    for angle_deg in [-60.0, -45.0, -30.0, 45.0]:
        _assert_spherical_recovery(angle_deg)


def _assert_spherical_recovery(angle_deg):
    shape = (65, 192, 81)
    step, pixel_size, sigma = 0.4, 0.168, 1.2
    theta = np.deg2rad(angle_deg)
    axial_step = step * np.sin(abs(theta))
    lateral_step = np.sign(theta) * step * np.cos(theta)
    z = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    x = np.arange(shape[2])[None, None, :]
    center = np.array(
        [
            (shape[0] - 1) / 2 * axial_step,
            (shape[1] - 1) / 2 * pixel_size + (shape[0] - 1) / 2 * lateral_step,
            (shape[2] - 1) / 2 * pixel_size,
        ]
    )
    # Sample a physical spherical Gaussian along the mechanical stage path.
    squared_distance = (
        (z * axial_step - center[0]) ** 2
        + (y * pixel_size + z * lateral_step - center[1]) ** 2
        + (x * pixel_size - center[2]) ** 2
    )
    raw = (220 * np.exp(-squared_distance / (2 * sigma**2))).astype(np.float32)
    calibration = janelia.resolve_deskew_calibration(_stage_scan_metadata(angle_deg))
    geometry = janelia.deskew_geometry(*shape[:2], calibration["shear_px_per_plane"])
    out = np.stack(list(janelia.deskew_planes(lambda index: raw[index], shape, geometry)))
    coordinates = [
        np.arange(out.shape[0]) * calibration["z_spacing_um"],
        (np.arange(out.shape[1]) - geometry["y_offset"]) * pixel_size,
        np.arange(out.shape[2]) * pixel_size,
    ]
    mass = out.sum(dtype=np.float64)
    measured_center = []
    variances = []
    for axis, coordinate in enumerate(coordinates):
        marginal = out.sum(axis=tuple(i for i in range(3) if i != axis), dtype=np.float64)
        mean = np.sum(marginal * coordinate) / mass
        measured_center.append(mean)
        variances.append(np.sum(marginal * (coordinate - mean) ** 2) / mass)
    yz_covariance = (
        np.sum(
            out.sum(axis=2, dtype=np.float64)
            * (coordinates[0] - measured_center[0])[:, None]
            * (coordinates[1] - measured_center[1])[None, :]
        )
        / mass
    )
    np.testing.assert_allclose(measured_center, center, atol=0.01)
    np.testing.assert_allclose(variances, sigma**2, rtol=0.04)
    assert abs(yz_covariance) < 0.02 * sigma**2


def test_deskew_straightens_yz_without_mixing_z_planes():
    geometry = janelia.deskew_geometry(3, 4, -1.0)
    assert geometry["shape_zy"] == (3, 6)
    assert geometry["y_offset"] == 2.0
    raw = np.zeros((3, 4, 3), np.float32)
    for z in range(3):
        raw[z, 1 + z, 1] = 100
    calls = []

    def get_plane(z):
        calls.append(z)
        return raw[z]

    out = np.stack(list(janelia.deskew_planes(get_plane, raw.shape, geometry)))
    assert out.shape == (3, 6, 3)
    assert np.array_equal(out[:, 3, 1], np.full(3, 100, np.uint8))
    assert not out[:, :, (0, 2)].any()
    assert calls == list(range(3))


def test_existing_group_rejects_rotated_or_different_transform():
    with tempfile.TemporaryDirectory() as work:
        dest = Path(work) / "deskew.zarr"
        dest.mkdir()
        attrs = {"deskew": {"shear_px_per_plane": -2.0}}
        (dest / ".zattrs").write_text(json.dumps(attrs))
        try:
            janelia.check_existing_deskew(dest, -2.0, 0.28)
        except ValueError as exc:
            assert "--force" in str(exc)
        else:
            raise AssertionError("the rotated group was reused")
        attrs["deskew"].update(method="projected_stage_scan_y_shear", z_spacing_um=0.28)
        (dest / ".zattrs").write_text(json.dumps(attrs))
        janelia.check_existing_deskew(dest, -2.0, 0.28)
        try:
            janelia.check_existing_deskew(dest, -2.381, 0.28)
        except ValueError:
            pass
        else:
            raise AssertionError("an existing group hid a changed coefficient")
        with np.testing.assert_raises_regex(ValueError, "--force"):
            janelia.check_existing_deskew(dest, -2.0, 0.4)


def test_deskew_interpolates_y_and_normalizes_edge_support():
    geometry = janelia.deskew_geometry(2, 2, -0.5)
    raw = np.array([[[100], [200]], [[20], [40]]], np.float32)
    out = np.stack(list(janelia.deskew_planes(lambda z: raw[z], raw.shape, geometry)))
    assert out.shape == (2, 3, 1)
    assert out[0, 0, 0] == 100
    assert out[0, 1, 0] == 150
    assert out[0, 2, 0] == 200
    assert out[1, 0, 0] == 20
