# Janelia ASLM YZ deskew (`janelia_n5.py`)

`janelia_n5.py` converts a Navigate / BigDataViewer N5 tile into uint8
OME-Zarr pyramids. `_sheared` preserves the recorded camera planes. `_deskew`
shifts those planes along Y, retaining the full tile without an extra rotation.

## Geometry correction

For `20260918_janelia_brain_big_stitch_488nm_22x/36x_001`, the metadata records:

- 500 Z planes, 648 camera Y rows, and 2048 X columns.
- A **mechanical stage step** of 0.4 um, with 0.168 um camera pixels.
- Ordinary `z-stack` acquisition, `primary_z_axis=z`, `waveform_template=Default`,
  and no focus displacement during the scan.
- A YZ angle of 315 degrees, equivalent to -45 degrees, and rotation disabled.
- XML plane-normal Z spacing of 0.282842712 um.

The corrected model projects the mechanical step into camera coordinates:

```text
dz = step * sin(abs(angle))          = 0.282842712 um
dy = sign(angle) * step * cos(angle) = -0.282842712 um
s  = dy / camera_pixel_y            = -1.683587574 pixels/plane
```

This uses the stage-scan convention in PetaKit's `deskewFrame3D` and
`XR_deskewRotateFrame`, without their optional rotation. The code checks that
the projected Z spacing agrees with the XML. It rejects unsupported acquisition
modes instead of silently applying this convention to an optical-step or
constant-velocity scan.

The XML Y coefficient, -2.380952, was computed from the **unprojected** 0.4 um
step. Combining it with either the projected XML Z spacing or an unprojected
0.4 um output Z spacing gives a different geometry. In particular, the previous
ClearEx-style candidate used 0.4 um displacement along both Y and Z: a resultant
step of 0.565685 um, not 0.4 um. The earlier shear-plus-rotation output also
interpolated across camera planes and degraded XY. Neither is the default now.

This interpretation is supported by the recorded acquisition mode and current
Navigate/PetaKit implementations, but has **not** been calibrated against beads
or a reference export from Kevin. It is not a claim of numerical equivalence to
ClearEx: its Navigate adapter infers a stage-Y scan and changes the spacing used
by its affine. Kevin's stage-axis mapping concerns tile placement metadata;
this script still processes one tile, not a stitched stage-coordinate volume.

## Resampling and coverage

```text
z_out = z
y_out = y + s*z + offset
x_out = x
offset = max(0, -s*(nz-1))
out_ny = ny + ceil(abs(s)*(nz-1))
```

Each output Z plane samples exactly one camera plane with linear Y interpolation.
Partial edge support is normalized, so boundary rows do not become artificially
dim. Bounds use `ceil` to retain fractional trailing coverage. The raw intensity
mapping is applied before interpolation; rounding to uint8 happens once afterward.
There is no background subtraction, deconvolution, or artificial tissue filling.

The full scan shifts by about 840 Y pixels, more than the 648-row camera height.
Its measured support therefore remains a parallelogram inside a rectangular
array. Black corners are expected with full-tile preservation; removing all of
them would discard part of the scan. They are separate from distortion or blur
**within** the measured tissue.

## Metadata and verification

Spacings are divided by the recorded 22x expansion factor. `36x` is the objective.

| group | shape (Z,Y,X), setup 0 | spacing (Z,Y,X), biological nm |
|---|---|---|
| `_sheared` | 500 x 648 x 2048 | 12.8565, 7.6364, 7.6364 |
| `_deskew` | 500 x 1489 x 2048 | 12.8565, 7.6364, 7.6364 |

Deskew metadata records the effective and XML coefficients, angle, projected
spacing, offset, interpolation, and source-to-output pixel affine. An existing
group with another method, coefficient, or Z spacing requires `--force`.
`--shear-px-per-plane` overrides only the lateral shift, not the Z calibration.

Tests simulate spherical objects acquired along an oblique mechanical stage path
and verify recovered centroids, physical variances, and YZ covariance. They also
cover angle signs, invalid calibration, stale outputs, and fractional edge support.

For visual checks, read **the actual resampled output**, compare corresponding
source points, use the recorded physical spacing, and keep intensity limits fixed.
Do not stretch orthogonal panels with `aspect="auto"`. Directly inverse-mapping
raw data to a display grid interpolates along the oblique raw Z direction and can
create diagonal fine streaks absent from the native deskewed grid. Neither such
previews nor apparent roundness of tissue alone is a geometry calibration.

## Usage

```bash
conda run -n pytc python tutorials/liconn_ingest/janelia_n5.py \
  /data/donglai/liconn/janelia_aslm/raw/20260918_janelia_brain_big_stitch_488nm_22x/36x_001 \
  --setup 0 --work /data/donglai/liconn/janelia_aslm/processed \
  --only deskew --force

conda run -n pytc python tutorials/liconn_ingest/run_tests.py
```

`--only deskew` leaves `_sheared.zarr` untouched. Without `--upload`, processing
stays local. Reload an existing viewer layer after replacing its Zarr group:
the new shape and spatial metadata differ from the old export.

## Batch processing with Slurm

`janelia_batch.py plan` discovers XML setups, validates their calibration and
N5 plane files, and writes a fixed task manifest. Metadata-only N5 setup
placeholders are ignored. The current raw directory contains 16 acquisitions
and 166 populated setups. Each task writes both `_sheared` and `_deskew`, reusing
existing compatible groups without `--force` or uploading anything.

On `cajal`, the sole partition is `gpu`, but these jobs request **no GPU**.
Use 4 CPUs and 16 GB per task, with eight concurrent tasks. Codec and numerical
thread pools are limited to avoid oversubscription. For example, from the repo:

```bash
RUN=/data/donglai/liconn/janelia_aslm/processed/_slurm/$(date -u +%Y%m%dT%H%M%SZ)
conda run -n pytc python tutorials/liconn_ingest/janelia_batch.py plan \
  --raw /data/donglai/liconn/janelia_aslm/raw \
  --work /data/donglai/liconn/janelia_aslm/processed --manifest "$RUN/tasks.json"
mkdir -p "$RUN/logs"
# Use the manifest task count minus one for the upper array index.
JOB=$(sbatch --parsable --array=0-165%8 \
  --output="$RUN/logs/%A_%a.out" --error="$RUN/logs/%A_%a.err" \
  tutorials/liconn_ingest/slurm/janelia_ingest.sbatch "$RUN" "$PWD")
sbatch --dependency="afterany:$JOB" --job-name=janelia_verify \
  --cpus-per-task=1 --mem=2G --time=00:10:00 \
  --output="$RUN/logs/verify_%j.out" --error="$RUN/logs/verify_%j.err" \
  tutorials/liconn_ingest/slurm/janelia_ingest.sbatch "$RUN" "$PWD" verify
```

Do not submit overlapping arrays against the same output groups. Supply a saved
source tree as `CODE_ROOT` (the second script argument) to isolate a run from
subsequent code edits; the submitted September 28 run does this under `RUN/code`.

Per-task records persist in `RUN/status/<index>.json`, including failures and
Slurm job IDs. The `afterany` verification job runs even if conversion fails,
checks both outputs' level-0 shape, dtype, spacing, and deskew calibration, and
writes `RUN/summary.json`. A successful summary has `ok: true`. This is important
on this host because Slurm accounting is disabled. Verification checks metadata
and normal worker completion, not an exhaustive read of every output chunk.

## References

- [PetaKit deskewFrame3D](https://github.com/abcucberkeley/PetaKit5D/blob/f7dbfd5cc5947a32e6028a6c7b895ff0b891c673/pointDetection/llsmtools/transforms/deskewFrame3D.m): lateral stage projection and full-volume bounds.
- [PetaKit XR_deskewRotateFrame](https://github.com/abcucberkeley/PetaKit5D/blob/f7dbfd5cc5947a32e6028a6c7b895ff0b891c673/microscopeDataProcessing/deskew_rotate/XR_deskewRotateFrame.m): projected axial spacing.
- [Navigate](https://github.com/TheDeanLab/navigate): `src/navigate/model/features/common_features.py` stage stepping and `src/navigate/model/metadata_sources/bdv_metadata.py` XML generation.
- [ClearEx](https://github.com/TheDeanLab/clearex): `src/clearex/io/experiment.py` Navigate geometry inference and `src/clearex/shear/pipeline.py` affine spacing, bounds, and support normalization.
