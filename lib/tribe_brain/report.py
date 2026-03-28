"""Multi-page A4 PDF export for brain surfaces and optional video thumbnails."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from neuralset.segments import Segment
from tribev2.plotting import PlotBrain
from tribev2.plotting.utils import (
    get_clip,
    get_text,
    has_video,
    robust_normalize,
)


def normalize_preds_like_timeline(
    preds: np.ndarray,
    norm_percentile: float | None,
) -> tuple[np.ndarray, bool]:
    """Match TRIBE ``plot_timesteps`` global robust norm over all time-by-vertex values.

    If ``norm_percentile`` is set, returns normalized data and True so ``plot_surf`` skips
    per-panel ``norm_percentile``. Otherwise returns the array unchanged and False.
    """
    arr = np.asarray(preds)
    if norm_percentile is None:
        return arr, False
    return robust_normalize(arr, percentile=norm_percentile), True


def segment_video_frame(segment: Segment, *, try_video: bool):
    """First RGB frame of the segment clip, or None if unavailable or on error."""
    if not try_video:
        return None
    if not has_video(segment):
        return None
    clip = get_clip(segment)
    if clip is None:
        return None
    try:
        return clip.get_frame(0)
    except (AttributeError, OSError, ValueError, RuntimeError):
        return None


def build_report_page_figure(
    plotter: PlotBrain,
    preds_batch: np.ndarray,
    segments_batch: list[Segment],
    *,
    page_idx: int,
    n_pages: int,
    video_name: str,
    views: str | list[str],
    cmap: str,
    norm_already_global: bool,
    norm_percentile: float | None,
    show_stimuli: bool,
    brains_per_page: int,
):
    """Build one A4 portrait figure with up to ``brains_per_page`` brain panels.

    Each panel may include a video thumbnail, ``plotter.plot_surf`` output, and a title with
    time range plus transcribed text for that window.
    """
    n = len(preds_batch)
    bpp = brains_per_page
    assert 1 <= n <= bpp

    plot_kw: dict = {"views": views, "cmap": cmap}
    if not norm_already_global and norm_percentile is not None:
        plot_kw["norm_percentile"] = norm_percentile

    if show_stimuli:
        if bpp == 2:
            mosaic = [["v0"], ["b0"], ["v1"], ["b1"]]
            height_ratios = [0.3, 1.0, 0.3, 1.0]
        else:
            mosaic = [
                ["v0", "v1"],
                ["b0", "b1"],
                ["v2", "v3"],
                ["b2", "b3"],
            ]
            height_ratios = [0.26, 1.0, 0.26, 1.0]
    else:
        if bpp == 2:
            mosaic = [["b0"], ["b1"]]
            height_ratios = [1.0, 1.0]
        else:
            mosaic = [["b0", "b1"], ["b2", "b3"]]
            height_ratios = [1.0, 1.0]

    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(8.27, 11.69),
        height_ratios=height_ratios,
        gridspec_kw={"hspace": 0.18, "wspace": 0.1},
    )

    for i in range(bpp):
        bi = f"b{i}"
        if i >= n:
            axes[bi].set_visible(False)
            if show_stimuli:
                axes[f"v{i}"].set_visible(False)
            continue

        seg = segments_batch[i]
        data = np.asarray(preds_batch[i])
        if show_stimuli:
            vi = f"v{i}"
            img = segment_video_frame(seg, try_video=True)
            if img is not None:
                axes[vi].imshow(img)
            axes[vi].axis("off")
            if img is None:
                axes[vi].set_visible(False)

        plotter.plot_surf(data, axes=axes[bi], **plot_kw)
        txt = get_text(seg, remove_punctuation=False)
        wrapped = textwrap.fill(txt, width=56) if txt else ""
        cap = f"{seg.start:.2f}–{seg.stop:.2f} s"
        if wrapped:
            cap = f"{cap}\n{wrapped}"
        if len(cap) > 1200:
            cap = cap[:1197] + "..."
        axes[bi].set_title(cap, fontsize=7, loc="left", pad=4)

    fig.suptitle(
        f"{video_name}\nTRIBE v2 — page {page_idx + 1} / {n_pages}",
        fontsize=11,
        y=0.995,
    )
    return fig


def export_brain_report(
    plotter: PlotBrain,
    preds_plot: np.ndarray,
    segments_plot: list[Segment],
    *,
    pdf_path: Path | None,
    png_first_page_path: Path | None,
    video_name: str,
    brains_per_page: int,
    views: str | list[str],
    cmap: str,
    norm_percentile: float | None,
    show_stimuli: bool,
    figure_dpi: int,
) -> None:
    """Write a multi-page PDF and/or a PNG of page 1 from predictions and segments.

    PNG-only mode renders just the first page. If ``get_frame`` fails with stimuli on, retries
    once without video thumbnails.
    """
    preds_use, norm_global = normalize_preds_like_timeline(
        preds_plot,
        norm_percentile,
    )
    n = int(preds_use.shape[0])
    if n == 0:
        return
    bpp = brains_per_page
    n_pages = (n + bpp - 1) // bpp
    only_first_png = pdf_path is None and png_first_page_path is not None
    n_pages_loop = 1 if only_first_png else n_pages

    def render_pages(stimuli_on: bool) -> None:
        first_saved = False
        pdf: PdfPages | None = None
        try:
            if pdf_path is not None:
                pdf = PdfPages(pdf_path)
            for p in range(n_pages_loop):
                sl = slice(p * bpp, min((p + 1) * bpp, n))
                batch_p = preds_use[sl]
                batch_s = segments_plot[sl]
                fig = build_report_page_figure(
                    plotter,
                    batch_p,
                    batch_s,
                    page_idx=p,
                    n_pages=n_pages,
                    video_name=video_name,
                    views=views,
                    cmap=cmap,
                    norm_already_global=norm_global,
                    norm_percentile=norm_percentile,
                    show_stimuli=stimuli_on,
                    brains_per_page=bpp,
                )
                if pdf is not None:
                    pdf.savefig(fig, dpi=figure_dpi, bbox_inches="tight")
                need_png = (
                    png_first_page_path is not None and not first_saved and p == 0
                )
                if need_png:
                    fig.savefig(
                        png_first_page_path,
                        dpi=figure_dpi,
                        bbox_inches="tight",
                    )
                    first_saved = True
                plt.close(fig)
        finally:
            if pdf is not None:
                pdf.close()

    try:
        render_pages(show_stimuli)
    except AttributeError as e:
        if show_stimuli and "get_frame" in str(e):
            print(
                "Warning: video stimuli failed for the report; "
                "retrying without thumbnails.",
                file=sys.stderr,
            )
            render_pages(False)
        else:
            raise
