#!/usr/bin/env python3
"""Regenerate the figures and numbers of the LICONN preprocessing tutorial.

Reads a few evenly spaced Z planes from one or two raw LICONN ND2 volumes,
runs each preprocessing step of ``connectomics.data.normalization`` and writes
``images/*.png`` plus ``images/metrics.json``. Only the sampled planes are
loaded, so a 6 GB ND2 needs about 200 MB of memory.

    python tutorials/liconn_preprocess/make_figures.py \
        --volume "ExPID96_2ndgel_S1_40XW001_18x Mojtaba Tavakoli.nd2" \
        --second "ExPID96_2ndgel_S4_40XW002_32x Mojtaba Tavakoli.nd2"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nd2  # noqa: E402
import numpy as np  # noqa: E402

from connectomics.data.normalization import (  # noqa: E402
    decode_aligned,
    display_from_codes,
    encode_aligned,
    fit_alignment,
    flat_field_image,
    to_aligned,
)
from connectomics.data.normalization.display import (  # noqa: E402
    DISPLAY_BLACK,
    DISPLAY_GAMMA,
    DISPLAY_WHITE,
)
from connectomics.data.normalization.noise import noise_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATS_PLANES = 16
CROP = 384


def _legacy_module():
    """scripts/preprocess_liconn.py, for the old clip + CLAHE baseline."""
    spec = importlib.util.spec_from_file_location(
        "preprocess_liconn", REPO / "scripts" / "preprocess_liconn.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SampledVolume:
    """ZYX view of an ND2 file that reads only the requested planes."""

    def __init__(self, path: Path):
        self.path = path
        self._file = nd2.ND2File(str(path))
        self._array = self._file.to_dask()
        self.shape = tuple(int(v) for v in self._array.shape)
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, z: int) -> np.ndarray:
        z = int(z)
        if z not in self._cache:
            self._cache[z] = np.asarray(self._array[z])
        return self._cache[z]

    def close(self) -> None:
        self._file.close()


def _save(fig, name: str) -> None:
    fig.savefig(HERE / "images" / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote images/{name}")


def _block_medians(image: np.ndarray, block: int) -> np.ndarray:
    rows, cols = image.shape[0] // block, image.shape[1] // block
    view = image[: rows * block, : cols * block].reshape(rows, block, cols, block)
    return np.median(view.transpose(0, 2, 1, 3).reshape(rows, cols, -1), axis=-1)


def _radial_profile(image: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices(image.shape)
    radius = np.hypot(y - image.shape[0] / 2, x - image.shape[1] / 2)
    radius /= radius.max()
    edges = np.linspace(0, 1, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    values = image.ravel()
    medians = np.array([np.median(values[index == i]) for i in range(bins)])
    return 0.5 * (edges[1:] + edges[:-1]), medians


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--volume", type=Path, required=True, help="Main raw ND2 volume.")
    parser.add_argument("--second", type=Path, help="Optional second raw ND2 volume.")
    parser.add_argument("--z", type=int, help="Plane shown in figures (default: middle).")
    args = parser.parse_args()
    (HERE / "images").mkdir(exist_ok=True)
    metrics: dict = {}

    volume = SampledVolume(args.volume)
    z_show = volume.shape[0] // 2 if args.z is None else args.z
    raw = volume[z_show].astype(np.float32)
    metrics["volume"] = args.volume.stem
    metrics["shape_zyx"] = list(volume.shape)
    metrics["z_shown"] = z_show

    # --- Step 0: why the old recipe had to change -------------------------------
    pct = np.percentile(raw, [1, 50, 99, 99.9])
    metrics["raw_percentiles_p1_p50_p99_p999"] = [float(v) for v in pct]
    metrics["old_clip_saturated_fraction"] = float(np.mean(raw > 350))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.logspace(np.log10(max(raw.min(), 50)), np.log10(raw.max() + 1), 160)
    ax.hist(raw.ravel(), bins=bins, color="0.25", histtype="step")
    ax.axvspan(120, 350, color="tab:blue", alpha=0.15, label="old clip 120-350")
    for value, label in zip(pct, ("p1", "p50", "p99", "p99.9")):
        ax.axvline(value, color="tab:red", ls="--", lw=0.8)
        ax.text(value, 0.02, label, rotation=90, fontsize=7,
                va="bottom", ha="right", color="tab:red", transform=ax.get_xaxis_transform())
    ax.set(xscale="log", yscale="log", xlabel="raw intensity (uint16)", ylabel="pixels",
           title=f"Raw histogram, z = {z_show}: "
           f"{100 * metrics['old_clip_saturated_fraction']:.1f}% above the old clip")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "01_raw_histogram.png")

    # --- Step A1: noise model ---------------------------------------------------
    alignment = fit_alignment(volume, stats_planes=STATS_PLANES)
    gain, offset = alignment["gain"], alignment["offset"]
    z_idx = alignment["stats_z_indices"]
    curve_means, curve_vars = noise_curve([volume[z] for z in z_idx])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(curve_means, curve_vars, "o", ms=4, label="flattest blocks, per brightness bin")
    grid = np.linspace(offset, curve_means.max() * 1.05, 50)
    ax.plot(grid, gain * (grid - offset), "k--", lw=1,
            label=f"variance = {gain:.2f} (x - {offset:.1f})")
    ax.set(xlabel="local mean (raw)", ylabel="noise variance (raw$^2$)",
           title=f"Shot-noise fit on {len(z_idx)} planes")
    ax.legend(fontsize=8)
    _save(fig, "02_noise_fit.png")
    metrics["gain"], metrics["offset"] = gain, offset
    metrics["tissue_fraction"] = alignment["tissue_fraction"]
    metrics["tissue_median_signal"] = alignment["tissue_median_signal"]

    # --- Step A2: flat field ----------------------------------------------------
    flat = flat_field_image(alignment, raw.shape)
    block = alignment["flat_field"]["block_size"]
    signal_blocks = np.median(
        np.stack([_block_medians(volume[z].astype(np.float32) - offset, block) for z in z_idx]),
        axis=0,
    )
    measured = signal_blocks / np.median(signal_blocks)
    fitted = _block_medians(flat, block)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, image, title, limits in (
        (axes[0], measured, "measured block signal", (0.3, 1.5)),
        (axes[1], fitted, "fitted exp-quadratic field", (0.3, 1.5)),
        (axes[2], measured / fitted, "ratio (measured / fit)", (0.5, 1.5)),
    ):
        im = ax.imshow(image, cmap="magma" if "ratio" not in title else "RdBu_r",
                       vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "03_flat_field.png")
    metrics["flat_field_corner"] = float(flat[:64, :64].mean())

    aligned = to_aligned(raw, alignment, flat)
    unflat = (raw - offset) / alignment["tissue_median_signal"]
    radius, prof_off = _radial_profile(unflat)
    _, prof_on = _radial_profile(aligned)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot(radius, prof_off, label="without flat field")
    ax.plot(radius, prof_on, label="with flat field")
    ax.axhline(1, color="0.6", lw=0.8)
    ax.set(xlabel="distance from tile center (0 = center, 1 = corner)",
           ylabel="median aligned intensity r", ylim=(0, None),
           title="Light compensation")
    ax.legend(fontsize=8)
    _save(fig, "04_radial_profile.png")

    # --- Step B: squeeze to the fixed uint8 table -------------------------------
    codes = encode_aligned(aligned)
    decoded = decode_aligned(codes)
    all_codes = np.arange(256)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(all_codes, decode_aligned(all_codes), color="tab:blue", label="decoded r")
    ax.set(xlabel="stored uint8 code", ylabel="aligned intensity r", yscale="log",
           ylim=(1e-3, 40))
    ax2 = ax.twinx()
    ax2.plot(all_codes, display_from_codes(all_codes.astype(np.uint8)), color="tab:orange",
             label="display brightness")
    ax2.set_ylabel("display (0-255)", color="tab:orange")
    ax.set_title("The two fixed curves")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    _save(fig, "05_codec_curves.png")

    qa = alignment["codec_qa"]
    metrics["codec_qa_sampled_planes"] = qa
    metrics["codes_used_z_shown"] = int(np.unique(codes).size)
    metrics["median_decoded_r"] = float(np.median(decoded))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    edges = np.linspace(-0.5, 6, 300)
    counts, _ = np.histogram(aligned.ravel(), bins=edges)
    ax.step(0.5 * (edges[1:] + edges[:-1]), counts / counts.sum() / np.diff(edges), "k",
            where="mid", label="raw-derived r")
    code_edges = decode_aligned(np.arange(257, dtype=np.float32) - 0.5).clip(0)
    code_counts = np.bincount(codes.ravel(), minlength=256)
    ax.plot(decode_aligned(all_codes), code_counts / code_counts.sum()
            / np.maximum(np.diff(code_edges), 1e-6), "o-", ms=2, color="tab:orange",
            label="decoded from uint8 (one bin per code)")
    ax.axvline(0, color="tab:red", lw=0.8, ls="--")
    ax.set(yscale="log", xlim=(-0.5, 6), ylim=(1e-6, None), xlabel="aligned intensity r",
           ylabel="density",
           title=f"Round trip: {100 * qa['clipped_low']:.2f}% below offset go to code 0; "
           f"RMS error {qa['rms_error_over_noise']:.3f} x noise SD")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    _save(fig, "06_roundtrip_histogram.png")

    # --- Step C: decode before any resampling ------------------------------------
    y2, x2 = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2

    def pool(image):
        return image[:y2, :x2].reshape(y2 // 2, 2, x2 // 2, 2).mean(axis=(1, 3))

    right = pool(decoded)
    wrong = decode_aligned(pool(codes.astype(np.float32)))
    bias = (wrong - right) / np.maximum(right, 1e-3)
    tissue = right > 0.3
    metrics["code_averaging_bias_2x2"] = float(np.median(bias[tissue]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(bias[tissue].ravel() * 100, bins=200, range=(-15, 5), color="tab:purple")
    axes[0].set(xlabel="relative error of decode(mean(codes)) vs mean(decode(codes)) [%]",
                ylabel="2x2 blocks", title="Averaging codes biases every level low")
    axes[0].title.set_fontsize(9)
    axes[0].xaxis.label.set_fontsize(8)
    sl = np.s_[: CROP // 2, : CROP // 2]
    im = axes[1].imshow(100 * bias[sl], cmap="RdBu_r", vmin=-10, vmax=10)
    axes[1].set_title("same error on a crop [%]", fontsize=9)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    _save(fig, "07_decode_before_resample.png")

    # --- Visualization: old pipeline vs aligned display ---------------------------
    legacy = _legacy_module()
    old = legacy.preprocess_xy_plane(raw, clip_intensity_range=(120.0, 350.0), clip_limit=0.03)
    shown = display_from_codes(codes)
    center = np.s_[raw.shape[0] // 2 - CROP // 2: raw.shape[0] // 2 + CROP // 2,
                   raw.shape[1] // 2 - CROP // 2: raw.shape[1] // 2 + CROP // 2]
    corner = np.s_[:CROP, :CROP]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
    for row, (image, label) in enumerate(((old, "old: clip 120-350 + CLAHE"),
                                          (shown, "new: aligned codes + display shader"))):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"{label}\nwhole tile", fontsize=8)
        axes[row, 1].imshow(image[center], cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title("center crop", fontsize=8)
        axes[row, 2].imshow(image[corner], cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title("corner crop", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    _save(fig, "08_display_comparison.png")

    # --- Cross-volume check on a second real acquisition --------------------------
    if args.second is not None:
        other = SampledVolume(args.second)
        other_alignment = fit_alignment(other, stats_planes=STATS_PLANES)
        z_other = other.shape[0] // 2
        other_raw = other[z_other].astype(np.float32)
        other_codes = encode_aligned(to_aligned(other_raw, other_alignment))
        other_aligned = decode_aligned(other_codes)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        name_a = args.volume.stem.split(" ")[0]
        name_b = args.second.stem.split(" ")[0]
        raw_bins = np.linspace(0, 900, 300)
        axes[0].hist(raw.ravel(), bins=raw_bins, histtype="step", density=True, label=name_a)
        axes[0].hist(other_raw.ravel(), bins=raw_bins, histtype="step", density=True,
                     label=name_b)
        axes[0].set(xlabel="raw intensity", ylabel="density", title="raw")
        # One bin per code (edges halfway between codes), so the discrete
        # codes do not alias into a comb.
        edges = decode_aligned(np.arange(257, dtype=np.float32) - 0.5).clip(0)
        for stored, label in ((codes, name_a), (other_codes, name_b)):
            counts = np.bincount(stored.ravel(), minlength=256)
            density = counts / counts.sum() / np.maximum(np.diff(edges), 1e-6)
            axes[1].plot(decode_aligned(np.arange(256)), density, label=label)
        axes[1].set(xlabel="decoded aligned intensity r", ylabel="density", xlim=(0, 5),
                    title="after alignment + codec (one bin per code)")
        for ax in axes:
            ax.legend(fontsize=7)
        _save(fig, "09_cross_volume.png")
        metrics["second_volume"] = {
            "volume": args.second.stem,
            "shape_zyx": list(other.shape),
            "z_shown": z_other,
            "gain": other_alignment["gain"],
            "offset": other_alignment["offset"],
            "tissue_fraction": other_alignment["tissue_fraction"],
            "tissue_median_signal": other_alignment["tissue_median_signal"],
            "raw_median": float(np.median(other_raw)),
            "median_decoded_r": float(np.median(other_aligned)),
            "codec_qa_sampled_planes": other_alignment["codec_qa"],
        }
        other.close()

    metrics["raw_median"] = float(np.median(raw))
    metrics["display"] = {"black": DISPLAY_BLACK, "white": DISPLAY_WHITE, "gamma": DISPLAY_GAMMA}
    (HERE / "images" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    volume.close()


if __name__ == "__main__":
    main()
