# De-shearing Janelia ASLM tiles (`janelia_n5.py`)

How [`janelia_n5.py`](janelia_n5.py) turns one tile of a Navigate / BigDataViewer
N5 acquisition (Kevin Dean lab, `coding/data/janelia_aslm/`) into a de-sheared
uint8 OME-Zarr pyramid. It covers the geometry, where each number comes from,
what was checked, and what is still open.

## 1. Why the raw data is sheared

These are **stage-scan** acquisitions. The light sheet sits at 45° to the stage
scan axis. Each camera frame is one oblique plane, and between frames the stage
moves by a fixed step (`experiment.yml: MicroscopeState.step_size = 0.4 µm`).

Stacked naively as (Z = frame index, Y, X), a structure that is compact in the
sample does not stay at the same Y from frame to frame. It drifts by a constant
amount per frame. In a raw Y–Z cross-section, compact objects appear as
diagonal streaks.

De-shearing undoes that drift. It is a Y translation that grows linearly with
the frame index.

## 2. The transform

For source plane `z` (0-based) and source row `y`:

```
y_out = y + a·z + offset        a      = shear, px per plane (negative here)
                                offset = |a|·(nz − 1)   when a < 0, so y_out ≥ 0
x_out = x
z_out = z
```

- The output height is `ny + ceil(|a|·(nz−1)) + 1`. For a 648-row, 500-plane
  tile with `a = −2.380952`, that is **1838 rows**.
- Each output plane depends on **one** source plane only. The transform is
  therefore streamed plane by plane, and memory use does not grow with Z.
- The fractional shift uses **linear interpolation** between the two nearest
  output rows (`shift_y`). Output outside the tilted region is filled with 0.
- The result is a parallelogram of data inside a zero-padded box. Plane 0 is
  at the bottom (offset ≈ 1188 rows) and the last plane is at the top.

Only a shear is applied: **no rotation** to the coverslip frame and no
isotropic resampling. The output axes are the stage-scan frame described in
§4.

## 3. Where `a` comes from (read, not chosen)

The shear is read per tile from `CH00_000000.xml`, in the setup's
`<ViewRegistration>`:

```xml
<ViewTransform type="affine" Name="Shearing Transform">
  <affine>1 0 0 0   0 1 -2.380952 0   0 0 1 0</affine>
</ViewTransform>
```

- This affine is written row-major in (x, y, z) pixel coordinates.
- Row y, column z holds `a = −2.380952`, which means `y' = y − 2.380952·z`.
- The value is `−step / xy_px = −0.4 / 0.168`: the full stage step, expressed
  in lateral pixels.

**ClearEx agrees.** Kevin's instruction was `shear_yz_deg = −45`. ClearEx
builds its shear in physical µm as `y' = y + tan(θ)·z`, so θ = −45° gives a
coefficient of −1. For a Navigate experiment with `BDVParameters.shear.shear_data: true`
and `shear_dimension: YZ` (both set here), ClearEx treats the data as stage-scan
and sets the scan-axis spacing to `scan_step / |coefficient| = 0.4 µm`.

- References:
  - `clearex/shear/pipeline.py::_resolve_affine_voxel_size_um_zyx`
  - `clearex/io/experiment.py::_infer_navigate_oblique_geometry`
- The result is `−1 × 0.4 µm = −0.4 µm = −2.381 px` per plane, the same as the
  XML value.

So the acquisition software's XML and the lab's own processing tool describe
the same transform. The script uses the XML value directly.

### The value we did not use

A PetaKit5D-style stage-scan deskew would use `step·cos45° / xy_px = 1.684 px`
per plane, with plane spacing `step·sin45° = 0.2828 µm`. That 0.2828 µm is the
`voxelSize` z the XML itself lists. The two conventions describe different
frames: plane-normal spacing versus stage-step spacing. We followed Kevin's
tools, which agree with each other.

### Empirical check (inconclusive)

The shift per plane was measured directly by phase-correlating frame `z`
against frame `z+k`, after two steps:

- subtracting the per-pixel z-median, which removes camera fixed-pattern noise;
- band-pass filtering the frames.

Results were a median of **1.88–2.07 px per plane** (tiles 0 and 20,
k = 4–16). That lies between 1.684 and 2.381, so it does not decide between
them. Oriented tissue (fibres) biases the estimator. The sign agrees with the
XML: content moves to larger Y as z increases, so the de-shear subtracts `|a|·z`.

Visually, the de-sheared Y–Z cross-section turns the diagonal streaks into
compact, rounded structures. Confirming the magnitude properly needs a bead
or point-source acquisition, or a check with Kevin.

## 4. Voxel spacing written to `.zattrs`

Spacings are divided by the expansion factor, which is parsed from
`experiment.yml: Saving.label_488nm` (`"22x"` → 22). The `36x` in the folder
names is the objective, not the expansion.

| group | shape (Z,Y,X), tile 0 | z | y, x | meaning of z |
|---|---|---|---|---|
| `_sheared` | 500 × 648 × 2048 | 0.2828 µm / 22 = 12.86 nm | 0.168 µm / 22 = 7.64 nm | XML voxel z (plane-normal spacing); **not** a rectilinear volume |
| `_deskew` | 500 × 1838 × 2048 | 0.4 µm / 22 = 18.18 nm | 7.64 nm | stage step, matching the XML/ClearEx frame |

`spacing_basis: biological` and `fold_source: experiment.yml:Saving.label_488nm`
are recorded, along with:

- `xml_shear_px_per_plane`;
- `stage_step_um`;
- the tile's `grid_translation_xyz_px`, for later stitching;
- a `deskew` block with `shear_px_per_plane`, `offset_px`, `interp` and `fill`.

## 5. Order relative to intensity normalization

```
raw uint16 frame ──► fit/apply intensity map ──► float 0–255 ─┬─► rint → uint8 → _sheared
                                                               └─► shift_y → rint → uint8 → _deskew
```

- **The intensity map is fitted on raw frames.** The default is `gat`: the
  noise-model Anscombe transform from `connectomics/data/normalization/anscombe.py`,
  loaded by file path in the same way `nd2_source.py` does. The fit uses 16
  evenly spaced *raw* frames, because the shot-noise model
  `var = gain·(x − offset)` holds in acquisition geometry. Interpolation would
  mix neighbouring pixels and shrink the variance.
- **The de-shear resamples the pre-rounding float.** Rounding happens once, so
  `_deskew` is a pure geometric resample of `_sheared` and carries the same
  intensity mapping.

## 6. Output and publishing

Both groups use `zarr_writer.write_pyramid`, the same layout as the
`liconn/moe` groups: zarr v2, 128³ chunks, blosc/zstd, and a 5-level pyramid
with floor halving. They are published to:

```
gs://donglai_public/liconn/janelia/<date_dataset>_<acq>/image/<name>_setupNN_{sheared,deskew}.zarr
```

`--upload` runs `gcloud storage rsync --delete-unmatched-destination-objects`
with the throttled settings from `ingest.py`, then runs the completeness check.

**Known false alarm.** zarr does not write chunks that equal the fill value, so
the empty corners of `_deskew` never exist as objects. `remote_group_complete`
expects every chunk, and it reports tile 0 as `649/1121 BAD`. Compare local
and remote object counts instead; for tile 0 they match exactly.

## 7. What this does not do

- **Stitching.** Each tile is de-sheared independently. Tile placement lives in
  the XML "Translation to Regular Grid" transform, which is recorded, and in
  `multi_positions.yml`. Neither is applied yet.
- **Rotation to the coverslip frame.** ClearEx's `auto_rotate_from_shear` would
  rotate by −atan(shear). It is off here, and off by default in ClearEx too.
- **Integer-pixel shortcut.** Kevin plans to make the shear an integer number
  of pixels in future acquisitions. When that happens, `shift_y` reduces to a
  pure translation with no interpolation.

## Usage

```bash
python tutorials/liconn_ingest/janelia_n5.py \
    <data>/janelia_aslm/20260918_janelia_brain_big_stitch_488nm_22x/36x_001 \
    --setup 0 --work <data>/janelia_aslm_work \
    [--only deskew] [--intensity-mode gat|linear] [--noise-model GAIN OFFSET] \
    [--force] [--upload]
```
