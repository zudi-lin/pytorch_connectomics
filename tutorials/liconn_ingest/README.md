# LICONN ingest — Drive drop → uint8 OME-Zarr pyramid → `gs://donglai_public`

The stage that sat in front of [`neuron_liconn_moe/`](../neuron_liconn_moe/README.md)
and was never in the repo. That tutorial starts from a published image group;
**this produces those.**

```
fetch → preprocess → publish → verify → prune        (or: run)
inspect · check · refold                             (out-of-band)
```

Each stage is idempotent, writes `provenance.json`, and drops a `COMPLETE.<stage>`
marker. `prune` refuses to delete anything the remote has not confirmed.

## Where each stage runs, and why there is a Docker image at all

`preprocess` needs `nd2` and `zarr`. Neither is in the `pytc` conda env and
neither is on the driving host — that gap is the entire reason this image
exists. `fetch` (rclone) and `publish`/`verify` (gcloud) stay on the host,
where the Drive and GCS credentials already live; containerising them would
mean mounting two credential stores to save nothing.

```bash
docker build --network=host -f tutorials/liconn_ingest/Dockerfile \
    -t pytc:liconn-ingest .
```

510 MB, CPU-only, self-contained. The earlier revision layered on `pytc:gpu`,
which meant an 18 GB CUDA image had to exist before a job that never touches a
GPU could run — and it did not exist on the machine this is driven from.
`--network=host` is needed because the default bridge network has no DNS here;
for the same reason the image installs no apt packages and proves the wheels
carry their own shared libraries with an import smoke test at build time.

## Configuration

`data.yaml` is **gitignored** (`tutorials/*/data.yaml`); `data.yaml.example` documents
the schema. Copy and fill. Precedence is CLI flag → `data.yaml` → built-in default.

**Source links do not go in it.** Drive share URLs are bearer credentials and this is a
public repository, where one mistaken `git add -f` publishes them irreversibly. They live
in the private research repo's own gitignored config,
`msi_liconn_deploy/data.yaml` under `sources:`; `data.yaml` here references a drop by its
`source_id` instead. A second copy would be both a second place to leak from and a second
place to drift.

A value of `FILL` is treated as absent, so it surfaces as the "set it" error rather than
propagating the literal string into a GCS path.

## Usage

```bash
# 0. one-time, interactive: rclone config   (Drive OAuth — not automatable)
rclone config                                   # create a remote named `gdrive`

# 1. Drive folder -> drop manifest. BY FOLDER ID, not name (see below).
python tutorials/liconn_ingest/ingest.py --inbox ./inbox \
    fetch --folder-id 1rq7jFpzyf2UoVa1vc8cuizkyUfHyHdqj --dry-run

# 2. what is actually in one of these files, before naming anything
python tutorials/liconn_ingest/ingest.py inspect ./inbox/ExPID71_120ms-30ms_600nm_40XW02.nd2

# 3. the whole drop: fetch each raw, convert, publish, verify, delete.
#    Skips anything already in the bucket, so an interrupted run resumes free.
python tutorials/liconn_ingest/ingest.py --inbox ./inbox --work ./work \
    run --clip-variant aligned_v1 --docker --prune
# --clip-variant, --work, --inbox and the bucket may instead come from data.yaml

# a specimen with no expansion fold anywhere, and two exposures per file
python tutorials/liconn_ingest/ingest.py --inbox ./expid71_snr --work ./work71snr \
    run --clip-variant aligned_v1 --fold-unknown --split-exposures \
        --docker --prune
```

`run` reads the `drop.json` that `fetch` wrote, and for each volume checks
`<dest>/.zattrs` before doing any work — so re-running after an interruption
costs one `gcloud storage ls` per volume and nothing else.

Single steps are there when you need them:

```bash
python tutorials/liconn_ingest/ingest.py --work ./work preprocess \
    ./inbox/ExPID99_32x_1.nd2 --clip-variant aligned_v1 \
    --drop-slug cerebellum --docker
python tutorials/liconn_ingest/ingest.py --work ./work publish ExPID99_32x_1_cerebellum
python tutorials/liconn_ingest/ingest.py --work ./work verify  ExPID99_32x_1_cerebellum
python tutorials/liconn_ingest/ingest.py --work ./work --inbox ./inbox \
    prune ExPID99_32x_1_cerebellum --dry-run
```

## The three out-of-band commands

`inspect <nd2>` — shape, dtype, optics spacing and per-channel exposures,
without converting anything. `run` uses it internally to name exposure-split
cubes before it has read a pixel.

`check <expid|gs://…>` — **read-only audit**: is each published group actually
complete? It computes the expected object count from the group's own metadata
(chunk grid per level, plus two metadata objects per level and two for the
group) and compares with a recursive listing. Needs no local copy, so it works
on volumes whose raw and scratch are long gone.

```bash
python tutorials/liconn_ingest/ingest.py check expid96
# OK   ExPID96_2ndgel_S1_40XW001_18x.zarr   1938/1938  ok
# BAD  ExPID71_Hippocampus_600nm_40XW02.zarr   312/1185  873 objects missing
```

This exists because *published* and *complete* are different things. An
interrupted `rsync` leaves a valid `.zattrs` with only some chunks, and a skip
test that asks only whether `.zattrs` exists will wave it through forever.
That is not hypothetical — it happened to three groups on 2026-09-21. `run`
now uses this check as its skip test, so a truncated group is re-published
rather than skipped.

`refold <name> --fold N [--fold-assumed]` — change a group's expansion fold
**without re-converting it**. The fold divides the optics spacing and never
touches a pixel, so it lives entirely in `.zattrs`; re-running `preprocess`
would redo ~30 min of CLAHE per volume to produce byte-identical chunks. Runs
on the host, no container.

## Bandwidth

Uncapped, `gcloud storage rsync` pushed 7 MB/s through this host's VPN tunnel
and froze the operator's SSH sessions — CPU, memory and disk all idle, the
uplink saturated, keystroke ACKs queued behind bulk transfer. rclone takes
`--bwlimit` (default `3M`). `gcloud storage` has no bandwidth flag, so it is
throttled by concurrency instead — `CLOUDSDK_STORAGE_PROCESS_COUNT=1`,
`CLOUDSDK_STORAGE_THREAD_COUNT=1`, set via environment so the operator's global
gcloud config is untouched. 1×1 is the floor; below that needs ADC (so rclone
can reach GCS with a real `--bwlimit`) or `tc`.

Run **one volume at a time**. Four concurrent CLAHE streams oversubscribed 8
cores and, worse, streamed ~200 GB through the page cache, evicting everything
interactive. `nice` does not help with that — only concurrency does.

## Intensity: `aligned` by default, legacy clip + CLAHE on request

**One uint8 copy serves both deep learning and the browser.** `--clip-mode
aligned` (the default) does the only per-volume step, alignment to a shared
profile, then stores through a table that is the same for every volume:

1. **Align** (per volume, fitted on 16 evenly spaced planes):
   `r = (x − offset) / (flat_field(y, x) · tissue_median)`. The offset and gain
   come from the shot-noise fit `variance = gain · (x − offset)`; the flat field
   is an exp-quadratic fitted to per-block tissue medians (vignetting); the
   tissue median uses every pixel of the tissue blocks (block median ≥ 3
   standard errors above the offset), so empty or resin regions do not move it
   and dim acquisitions keep all their tissue. `r` is 0 at no light and 1 at typical tissue, for
   any microscope, gain, exposure or labeling brightness.
2. **Store** `r` through the fixed code table
   `code = 255 · (√(clip(r, 0, 32) + 0.25) − 0.5) / (√32.25 − 0.5)`, where the
   square root follows the shot noise. A code means the same aligned intensity
   in every volume. r below 0 (under the fitted offset) becomes code 0; the
   fraction is logged and stored as `codec_qa_clipped_low`.
3. **Train** on decoded `r` with `data.image_transform.normalize: aligned-u8`,
   which decodes the table before any resize and computes no per-patch
   statistics (clip percentiles must stay 0 / 1). Per-patch min-max or z-score
   would undo the alignment. The rest of the nonlinearity
   is the model's job.
4. **View** the same codes through the fixed shader in `.zattrs`
   (`neuroglancer_shader`): black at r = 0.75, white at r = 4.5, γ = 1.5, so
   noise sits near black and the bright puncta stand out. No second copy and
   no histogram equalization.

Pyramid levels average in aligned units and re-encode
(`nd2_source.aligned_block_average`); averaging the square-root codes directly
would bias every coarser level low. `.zattrs` records the fit (offset, gain,
tissue median and tissue rule, tissue fraction, flat-field coefficients,
codec QA), the code table and the display. `provenance.json` keeps the
full fit under `alignment` and the `.zattrs` keys under `intensity_attrs`, so
`refold` can regenerate them without numpy. `--no-flat-field` skips the
vignetting correction.

Step-by-step walkthrough with figures:
[`../liconn_preprocess/README.md`](../liconn_preprocess/README.md).
Why it replaced the legacy recipe (evidence and figures in the Google Doc
"LICONN intensity normalization"): on ExPID96 the fixed 120–350 clip saturated
7.9% of pixels, the synapse/vesicle tail; per-plane p99 still flattens puncta
peaks; CLAHE raises sharpness at the defocused tile border to match the center
by amplifying noise; and per-plane statistics leave every volume on its own
scale.

**Aligned groups publish under `<expid>/image_aligned/`, not `<expid>/image/`.**
Names do not carry the recipe, so a shared folder would let `run` skip an
aligned conversion as "already published" and let `publish` overwrite the
clip+CLAHE group that `neuron_liconn_moe` reads. `check <expid> --image-dir
image_aligned` audits them. A work directory that already holds the other
recipe's conversion is refused unless you pass `--force` or use a separate
`--work`.

`--clip-mode percentile|fixed_intensity` reproduces the legacy published recipe
(per-plane clip, CLAHE with `--clip-limit`, default 0.03) and its exact
`.zattrs` key set. `clip_values` and `clip_limit` are refused with `aligned`,
including when left in `data.yaml`.

## The published layout is read, not chosen

Target: `gs://donglai_public/liconn/moe/<expid>/image/<name>.zarr` — the
five-level uint8 pyramid the existing groups use. `reference/published_group.json`
is the real metadata of `expid96/image/ExPID96_2ndgel_S1_40XW001_18x.zarr`,
recorded 2026-09-21, and `test_zarr_writer.py` asserts the writer reproduces it:
128³ chunks, `|u1`, blosc/zstd/clevel 5/shuffle 1, `.` dimension separator,
OME-NGFF 0.4 multiscales, and the same thirteen `.zattrs` keys
`scripts/preprocess_liconn.py` writes for the legacy recipe. A numcodecs default that moves fails the
build rather than the next 6 GB upload.

Names carry the drop folder (`ExPID99_32x_1_cerebellum`) and, when one Drive
folder holds two files of the same name, the first 8 characters of the Drive
file ID (`ExPID99_18x_2_cerebellum_1Byqvupl`). Both conventions were read back
off the published set. The previous revision of `ingest.py` wrote to
`<clip_variant>/zarr/<cube_id>.zarr`, which matches nothing in the bucket.

## Four things this tool will not do for you

**`--clip-variant` has no default.** It labels the intensity recipe in
provenance (e.g. `aligned_v1`). Every group under `image/` is
`clip_mode=percentile, clip_percentiles=[1,99]` plus CLAHE, while
`lessons/liconn_preprocessing.md` calls fixed `120–350` the reference recipe;
new conversions default to `aligned` (see above).

**It will not invent an expansion fold.** No fold in the stem and no `--fold`
means `preprocess` refuses. The fold is not in the ND2 either — searched
2026-09-21 across `custom_data` and `text_info` for expan/gel/fold/swell, zero
hits. `--fold-unknown` publishes raw optics spacing instead and stamps
`spacing_basis: optics_uncorrected` plus a `spacing_warning` into `.zattrs`,
because such a volume is **not** spatially comparable with the folded groups
and a log line does not travel with the data.

**It will not take the exposure from the filename.**
`ExPID71_120ms-30ms_600nm_40XW02.nd2` lists 120 first, but channel 0 is the
**30 ms** acquisition (`488_Low`) — the filename order is the reverse of the
channel order. `--split-exposures` reads the mapping from ND2 metadata and
cross-checks the *set* against the filename; if they disagree it publishes
neither. `naming.parse_stem` returns exposures as an unordered `frozenset` so
no caller can imply an order. (spec.md I7.)

**Drive filenames are not unique, and the folder is addressed by ID.** The
`Cerebellum` drop holds two *different* files both called `ExPID99_18x_2.nd2`
(9.93 GB and 11.86 GB), which `rclone copy` would silently collapse into one;
and the folder does not appear in `rclone lsd gdrive: --drive-shared-with-me`
at all, so a name-based fetch cannot reach it. `fetch --folder-id` plus one
`rclone backend copyid` per file handles both.

## Streaming, because the arithmetic says so

A 1199×2304×2304 uint16 ND2 is 12.7 GB and the driving host has ~24 GB free.
Level 0 is written one 128-plane chunk row at a time and each reduction reads
the level below off disk, so peak memory is a couple of chunk rows and does not
scale with Z. This is only equivalent to a whole-volume read because no
plane's output depends on another plane. `aligned` fits its one volume-wide
alignment from sampled planes **before** streaming starts; the legacy modes compute
their percentiles per plane. `test_nd2_source.py` pins both, since a statistic
that drifted during streaming would silently change the pixels.

The pixel math is the repo's, not a copy: `nd2_source.load_normalization_package`
loads `connectomics/data/normalization/` by path, and
`nd2_source.load_preprocess_module` loads `scripts/preprocess_liconn.py` behind a
stub for its one package-level import. The image stays free of torch and MONAI
without forking the pixel path.

## What is verified

- **58/58 tests pass** (host run, 2026-09-28; the image build re-runs them), and they gate the build
  (`python tutorials/liconn_ingest/run_tests.py`; the numpy-dependent modules
  skip loudly on a bare host).
- `write_pyramid` end to end on a synthetic 37×40×40 volume: level 0 comes
  back byte-identical, level 1 equals the block average of the aligned part of
  level 0, the odd trailing plane is dropped (37→18→9→4→2, the same
  floor-halving as the published 585→292→146→73→36), and every `.zarray`
  field matches the recorded published group.
- `fetch --dry-run` and `run` against the real Drive folders and the real
  bucket: the four `Cerebellum` names reproduce the four published group names
  exactly, and `run` skips them as already published.
- **The ND2 path is executed.** `nd2_source.iter_planes` was run on real drops
  on 2026-09-21. The dask lazy read was checked byte-identical to
  `ND2File.read_frame` at four z positions including the first and last, the
  optics spacing read from the file (`500 nm` z-step) independently matches the
  filename, and the per-channel exposure parse cross-checks against the
  structured channel names.

## Known rough edge

`nd2` emits `ND2File file not closed before garbage collection` when a caller
abandons `iter_planes` part-way (the `with` block only exits on full
iteration). Harmless for a completed run — every production call consumes the
whole generator — but it will appear in logs if you break out of the loop.
