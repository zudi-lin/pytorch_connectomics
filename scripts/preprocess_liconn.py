#!/usr/bin/env python3
"""Prepare a single-tile LICONN structural volume for PyTC/FFN.

The input is loaded at its native (level-0) resolution through PyTC's canonical
volume reader and mapped to uint8 by one of two intensity modes:

* ``aligned`` (default): one uint8 copy that serves both deep learning and
  browser viewing. Each volume is aligned to a shared intensity profile,
  ``r = (x - offset) / (flat_field(y, x) * tissue_median)`` (0 = no light,
  1 = tissue median, illumination falloff divided out), with offset and gain
  from a shot-noise fit on sampled planes. ``r`` is stored through the fixed
  code table in ``connectomics.data.normalization.codec`` (the same for every
  volume); training decodes it with ``image_transform.normalize: aligned-u8``
  and browsers apply the fixed display shader recorded in the attributes.
* ``linear``: each XY plane is clipped to a fixed raw range (120--350 by
  default) or per-plane percentiles, optionally CLAHE-enhanced, then scaled.

The processed volume can then be reduced by aligned ZYX arithmetic volume
averaging. For a LICONN level-0 image at 12x9x9 nm ZYX,
``--downsample-factor 2 2 2`` produces the 24x18x18 nm FFN scale.

ND2 inputs may contain multiple channels but must contain only one XY
position/tile; use ``--channel`` to select the structural channel. HDF5 output
is losslessly compressed by default. JPEG is not part of the scientific
preprocessing.

Example:
    python scripts/preprocess_liconn.py raw_structural.nd2 ffn_image.h5 \
        --channel 0 --downsample-factor 2 2 2 --input-spacing-nm 12 9 9
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from skimage import exposure

from connectomics.data.io import read_volume


DEFAULT_CLIP_INTENSITY_RANGE = (120.0, 350.0)
DEFAULT_STATS_PLANES = 16


def preprocess_xy_plane(
    plane: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    clip_intensity_range: tuple[float, float] | None = DEFAULT_CLIP_INTENSITY_RANGE,
    clip_limit: float | None = None,
) -> np.ndarray:
    """Clip one XY plane, optionally CLAHE-enhance it, and return uint8."""
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2D XY plane, got shape {plane.shape}.")
    if clip_intensity_range is None and not (
        0.0 <= lower_percentile < upper_percentile <= 100.0
    ):
        raise ValueError(
            "Percentiles must satisfy 0 <= lower < upper <= 100; "
            f"got {lower_percentile} and {upper_percentile}."
        )
    if clip_limit is not None and clip_limit <= 0.0:
        raise ValueError(f"CLAHE clip_limit must be positive, got {clip_limit}.")

    plane_float = np.asarray(plane, dtype=np.float32)
    if clip_intensity_range is None:
        low, high = np.percentile(plane_float, (lower_percentile, upper_percentile))
    else:
        low, high = (float(value) for value in clip_intensity_range)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Clipping bounds must be finite.")
    if high <= low:
        if clip_intensity_range is not None:
            raise ValueError(
                f"Fixed intensity range must satisfy low < high, got {low} and {high}."
            )
        return np.zeros(plane.shape, dtype=np.uint8)

    normalized = np.clip((plane_float - low) / (high - low), 0.0, 1.0)
    if clip_limit is not None:
        normalized = exposure.equalize_adapthist(normalized, clip_limit=clip_limit)
    return np.clip(np.rint(normalized * 255.0), 0.0, 255.0).astype(np.uint8)


def _validate_downsample_factor(
    factor: tuple[int, int, int], source_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Validate an aligned ZYX integer downsampling factor."""
    if len(factor) != 3:
        raise ValueError(f"Downsampling factor must contain Z, Y, and X; got {factor}.")
    factor = tuple(int(value) for value in factor)
    if any(value < 1 for value in factor):
        raise ValueError(f"Downsampling factors must be positive integers; got {factor}.")

    remainder = tuple(size % value for size, value in zip(source_shape, factor))
    if any(remainder):
        raise ValueError(
            "Source shape must be divisible by the aligned downsampling factor; "
            f"got shape {source_shape}, factor {factor}, and remainder {remainder}. "
            "Crop or pad explicitly before preprocessing."
        )
    return factor


def _validate_spacing(spacing_nm: tuple[float, float, float] | None) -> np.ndarray | None:
    """Return a validated ZYX spacing vector in nanometers."""
    if spacing_nm is None:
        return None
    if len(spacing_nm) != 3:
        raise ValueError(f"Input spacing must contain Z, Y, and X; got {spacing_nm}.")
    spacing = np.asarray(spacing_nm, dtype=np.float64)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError(f"Input spacing must contain positive finite values; got {spacing_nm}.")
    return spacing


def volume_average_uint8(volume: np.ndarray, factor: tuple[int, int, int]) -> np.ndarray:
    """Downsample a ZYX uint8 volume by aligned arithmetic block averaging.

    Block means are rounded to the nearest integer using NumPy's nearest-even
    rule. The input shape must be exactly divisible by the ZYX factor so that
    preprocessing never hides an implicit crop or padding policy.
    """
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError(f"Expected a ZYX uint8 volume, got shape {volume.shape}, {volume.dtype}.")
    factor = _validate_downsample_factor(factor, volume.shape)
    z_factor, y_factor, x_factor = factor
    z_size, y_size, x_size = volume.shape
    blocks = volume.astype(np.uint32).reshape(
        z_size // z_factor,
        z_factor,
        y_size // y_factor,
        y_factor,
        x_size // x_factor,
        x_factor,
    )
    averaged = blocks.mean(axis=(1, 3, 5), dtype=np.float64)
    return np.rint(averaged).astype(np.uint8)


def _select_structural_channel(volume: np.ndarray, channel: int) -> np.ndarray:
    """Return a ZYX structural volume from a PyTC-order input array."""
    if channel < 0:
        raise ValueError(f"Channel index must be non-negative, got {channel}.")
    if volume.ndim == 4:
        if channel >= volume.shape[0]:
            raise ValueError(
                f"Channel index {channel} is out of range for input shape {volume.shape}."
            )
        volume = volume[channel]
    elif volume.ndim in (2, 3):
        if channel != 0:
            raise ValueError(
                f"Input shape {volume.shape} has no channel axis; use --channel 0."
            )
    else:
        raise ValueError(
            f"Expected a 2D/3D volume or channel-first 4D volume, got shape {volume.shape}."
        )

    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    return volume


def preprocess_liconn(
    input_path: Path,
    output_path: Path,
    *,
    channel: int = 0,
    dataset: str = "main",
    intensity_mode: str = "aligned",
    stats_planes: int = DEFAULT_STATS_PLANES,
    noise_model: tuple[float, float] | None = None,
    flat_field: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    clip_intensity_range: tuple[float, float] | None = DEFAULT_CLIP_INTENSITY_RANGE,
    clip_limit: float | None = None,
    downsample_factor: tuple[int, int, int] = (1, 1, 1),
    input_spacing_nm: tuple[float, float, float] | None = None,
    compression: str | None = "gzip",
    compression_level: int = 4,
    overwrite: bool = False,
) -> Path:
    """Preprocess one LICONN volume and atomically write a uint8 HDF5 file.

    The intensity mapping is applied before downsampling; in ``aligned`` mode
    blocks are averaged in aligned units and re-encoded, never as codes.
    Downsampling is streamed over source Z planes so the full processed volume
    is never materialized in memory.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if output_path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"Output must be an HDF5 file (.h5 or .hdf5): {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists (pass --overwrite to replace it): {output_path}"
        )
    if not dataset or dataset.strip("/") == "":
        raise ValueError("HDF5 dataset name must not be empty.")
    if intensity_mode not in {"aligned", "linear"}:
        raise ValueError(
            f"Unknown intensity_mode {intensity_mode!r}; use 'aligned' or 'linear'."
        )
    if compression not in {None, "gzip", "lzf"}:
        raise ValueError(f"Unsupported HDF5 compression: {compression!r}.")
    if compression == "gzip" and not 0 <= compression_level <= 9:
        raise ValueError(f"Gzip compression level must be in [0, 9], got {compression_level}.")

    raw = read_volume(str(input_path))
    structural = _select_structural_channel(raw, channel)
    z_size, y_size, x_size = structural.shape
    downsample_factor = _validate_downsample_factor(downsample_factor, structural.shape)
    input_spacing = _validate_spacing(input_spacing_nm)
    z_factor, y_factor, x_factor = downsample_factor
    output_shape = (z_size // z_factor, y_size // y_factor, x_size // x_factor)
    output_z_size, output_y_size, output_x_size = output_shape

    if intensity_mode == "aligned":
        # Imported here, not at module level: tutorials/liconn_ingest/nd2_source.py
        # loads this script with `connectomics` stubbed down to `read_volume`.
        from connectomics.data.normalization import (
            alignment_attrs,
            encode_aligned,
            fit_alignment,
            flat_field_image,
            to_aligned,
        )

        alignment = fit_alignment(
            structural, stats_planes=stats_planes, noise_model=noise_model, flat_field=flat_field
        )
        flat = flat_field_image(alignment, (y_size, x_size))
        print(
            f"Alignment ({alignment['noise_model_source']} noise model): offset "
            f"{alignment['offset']:.1f}, gain {alignment['gain']:.3f}, tissue median "
            f"{alignment['tissue_median_over_gain']:.1f} x gain "
            f"({100 * alignment['tissue_fraction']:.0f}% tissue), flat field "
            f"{'min %.2f' % flat.min() if alignment['flat_field'] else 'off'}"
        )
        qa = alignment["codec_qa"]
        print(
            f"Codec QA (sampled planes): below offset {100 * qa['clipped_low']:.2f}%, "
            f"above r_max {100 * qa['clipped_high']:.3f}%, round-trip "
            f"{qa['rms_error_over_noise']:.3f} x noise SD"
        )

        def plane_values(plane: np.ndarray) -> np.ndarray:
            return to_aligned(plane, alignment, flat)

        def values_to_uint8(mean_values: np.ndarray) -> np.ndarray:
            return encode_aligned(mean_values)

        accumulate_dtype = np.float64
    else:

        def plane_values(plane: np.ndarray) -> np.ndarray:
            return preprocess_xy_plane(
                plane,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                clip_intensity_range=clip_intensity_range,
                clip_limit=clip_limit,
            )

        def values_to_uint8(mean_values: np.ndarray) -> np.ndarray:
            return np.rint(mean_values).astype(np.uint8)

        accumulate_dtype = np.uint64

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    compression_opts = compression_level if compression == "gzip" else None
    try:
        with h5py.File(temp_path, "w") as handle:
            output = handle.create_dataset(
                dataset,
                shape=output_shape,
                dtype=np.uint8,
                chunks=(1, min(1024, output_y_size), min(1024, output_x_size)),
                compression=compression,
                compression_opts=compression_opts,
                shuffle=compression is not None,
            )
            output.attrs["axes"] = "ZYX"
            output.attrs["source"] = str(input_path)
            output.attrs["source_channel"] = channel
            output.attrs["source_shape_zyx"] = structural.shape
            output.attrs["downsample_factor_zyx"] = downsample_factor
            output.attrs["downsample_method"] = (
                "aligned_intensity_volume_average"
                if intensity_mode == "aligned"
                else "aligned_arithmetic_volume_average"
            )
            output.attrs["downsample_rounding"] = "nearest_even"
            if input_spacing is not None:
                output.attrs["input_spacing_nm_zyx"] = input_spacing
                output.attrs["spacing_nm_zyx"] = input_spacing * np.asarray(
                    downsample_factor, dtype=np.float64
                )
            if intensity_mode == "aligned":
                for key, value in alignment_attrs(alignment).items():
                    output.attrs[key] = value
            else:
                if clip_intensity_range is None:
                    output.attrs["clip_mode"] = "percentile"
                    output.attrs["clip_percentiles"] = (lower_percentile, upper_percentile)
                else:
                    output.attrs["clip_mode"] = "fixed_intensity"
                    output.attrs["clip_intensity_range"] = clip_intensity_range
                output.attrs["clahe_clip_limit"] = (
                    "none" if clip_limit is None else clip_limit
                )

            block_voxels = z_factor * y_factor * x_factor
            for output_z in range(output_z_size):
                block_sum = np.zeros((output_y_size, output_x_size), dtype=accumulate_dtype)
                source_z_start = output_z * z_factor
                for z_offset in range(z_factor):
                    source_z = source_z_start + z_offset
                    values = plane_values(structural[source_z])
                    plane_blocks = values.astype(accumulate_dtype).reshape(
                        output_y_size,
                        y_factor,
                        output_x_size,
                        x_factor,
                    )
                    block_sum += plane_blocks.sum(axis=(1, 3), dtype=accumulate_dtype)

                    planes_processed = source_z + 1
                    if planes_processed % 25 == 0 or planes_processed == z_size:
                        print(f"Processed source XY planes: {planes_processed}/{z_size}")

                output[output_z] = values_to_uint8(block_sum / block_voxels)

        os.replace(temp_path, output_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map a single-tile LICONN structural volume to uint8 (shared aligned-"
            "intensity profile and fixed code table by default), optionally reduce it "
            "by aligned volume averaging, then save losslessly compressed uint8 HDF5."
        )
    )
    parser.add_argument("input", type=Path, help="Input volume (typically a single-tile ND2 file).")
    parser.add_argument("output", type=Path, help="Output .h5 or .hdf5 path.")
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Zero-based structural channel index for multi-channel input. Default: 0.",
    )
    parser.add_argument(
        "--dataset", default="main", help="Output HDF5 dataset name. Default: main."
    )
    parser.add_argument(
        "--intensity-mode",
        choices=("aligned", "linear"),
        default="aligned",
        help=(
            "aligned: per-volume alignment to the shared profile, fixed uint8 code "
            "table (default). linear: per-plane clip (+ optional CLAHE)."
        ),
    )
    parser.add_argument(
        "--stats-planes",
        type=int,
        default=DEFAULT_STATS_PLANES,
        help="aligned: evenly spaced Z planes used for the volume statistics. Default: 16.",
    )
    parser.add_argument(
        "--no-flat-field",
        dest="flat_field",
        action="store_false",
        help="aligned: skip the illumination (vignetting) correction.",
    )
    parser.add_argument(
        "--noise-model",
        type=float,
        nargs=2,
        metavar=("GAIN", "OFFSET"),
        default=None,
        help="aligned: use variance = GAIN * (x - OFFSET) instead of fitting it.",
    )
    clipping = parser.add_mutually_exclusive_group()
    clipping.add_argument(
        "--clip-intensity-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_CLIP_INTENSITY_RANGE,
        help=(
            "linear: fixed raw intensity clipping range applied to every XY plane. "
            "Default: 120 350."
        ),
    )
    clipping.add_argument(
        "--clip-percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help=(
            "linear: per-plane percentile clipping instead of the fixed range. "
            "Example: --clip-percentiles 1 99."
        ),
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=None,
        help="linear: skimage CLAHE clip limit, e.g. 0.03. Default: no CLAHE.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=(1, 1, 1),
        help=(
            "Aligned ZYX arithmetic volume-averaging factor after intensity mapping. "
            "Use 2 2 2 for LICONN 12x9x9 nm ZYX to FFN 24x18x18 nm. Default: 1 1 1."
        ),
    )
    parser.add_argument(
        "--input-spacing-nm",
        type=float,
        nargs=3,
        metavar=("Z", "Y", "X"),
        default=None,
        help="Optional input ZYX voxel spacing in nm; records input and output spacing.",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression. Default: gzip.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Gzip compression level. Default: 4.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    if args.clip_percentiles is None:
        lower_percentile, upper_percentile = (1.0, 99.0)
        clip_intensity_range = tuple(args.clip_intensity_range)
    else:
        lower_percentile, upper_percentile = args.clip_percentiles
        clip_intensity_range = None
    output_path = preprocess_liconn(
        args.input,
        args.output,
        channel=args.channel,
        dataset=args.dataset,
        intensity_mode=args.intensity_mode,
        stats_planes=args.stats_planes,
        flat_field=args.flat_field,
        noise_model=tuple(args.noise_model) if args.noise_model is not None else None,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        clip_intensity_range=clip_intensity_range,
        clip_limit=args.clip_limit,
        downsample_factor=tuple(args.downsample_factor),
        input_spacing_nm=(
            tuple(args.input_spacing_nm) if args.input_spacing_nm is not None else None
        ),
        compression=compression,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
    )
    print(f"Wrote uint8 HDF5 volume: {output_path}")


if __name__ == "__main__":
    main()
