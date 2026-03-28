"""
CLI entrypoint: video file -> Meta TRIBE v2 cortical predictions -> multi-page PDF report.

Upstream model: https://huggingface.co/facebook/tribev2

Requires Hugging Face auth for weights; Llama 3.2 (text encoder) is gated — approve access on HF.
Default transcription path uses ``uvx whisperx``; install ``uv`` in the venv or use ``--audio-only-events``.

Example::

    python scripts/ad_brain.py path/to/video.mp4 --out-dir ./out --device mps
"""

from __future__ import annotations
from lib.tribe_brain.support import (
    apply_whisper_cpu_float32_patch,
    events_from_video,
    extractor_config_update,
    is_hf_gated_llama_access_error,
    parse_views,
    prepare_timeline_for_plot,
    print_llama_gated_help,
    resolve_device,
)
from lib.tribe_brain.report import export_brain_report
from tribev2.plotting import PlotBrain
from tribev2 import TribeModel
from matplotlib import pyplot as plt
import numpy as np

import argparse
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")


def build_arg_parser() -> argparse.ArgumentParser:
    """Configure argparse with grouped options for I/O, model, transcription, and visualization."""
    p = argparse.ArgumentParser(
        description="Run TRIBE v2 on a video and export brain-surface visualizations.",
    )

    g_in = p.add_argument_group("Input / output")
    g_in.add_argument(
        "video",
        type=Path,
        help="Path to a video file (.mp4, .mov, .webm, …).",
    )
    g_in.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Output directory (created if needed). Default: ./out",
    )
    g_in.add_argument(
        "--cache-folder",
        type=Path,
        default=Path("cache"),
        help="Feature cache directory. Default: ./cache",
    )
    g_in.add_argument(
        "--format",
        dest="out_formats",
        default="pdf,png",
        help=(
            "Comma-separated: pdf, png. PDF writes the paginated report. "
            "Without --include-timeline, PNG is page 1 of the report only. "
            "With --include-timeline, PNG also writes the horizontal timeline strip."
        ),
    )

    g_model = p.add_argument_group("Model & device")
    g_model.add_argument(
        "--device",
        choices=("auto", "mps", "cpu", "cuda"),
        default="auto",
        help="PyTorch device: auto picks cuda, else mps (Apple Silicon), else cpu.",
    )
    g_model.add_argument(
        "--repo",
        default="facebook/tribev2",
        help="Hugging Face model id. Default: facebook/tribev2",
    )

    g_tx = p.add_argument_group("Transcription & events")
    g_tx.add_argument(
        "--audio-only-events",
        action="store_true",
        help="Skip WhisperX (no uvx). Keeps video/audio features; no word-level text.",
    )

    g_vis = p.add_argument_group("Visualization")
    g_vis.add_argument(
        "--no-stimuli",
        action="store_true",
        help="Omit video thumbnails and transcribed captions on figures.",
    )
    g_vis.add_argument(
        "--views",
        default="left",
        help="Cortical view(s), e.g. left or left,right (comma-separated).",
    )
    g_vis.add_argument(
        "--norm-percentile",
        type=float,
        default=None,
        metavar="P",
        help="Robust normalization percentile for maps. Default: raw scale.",
    )
    g_vis.add_argument(
        "--figure-dpi",
        type=int,
        default=240,
        metavar="D",
        help="Export DPI for rasterized parts (PyVista surfaces). Default: 240.",
    )
    g_vis.add_argument(
        "--max-timeline-columns",
        type=int,
        default=48,
        metavar="M",
        help=(
            "Max number of time windows shown in exports (subsamples display only). "
            "0 = all steps (heavy). Default: 48."
        ),
    )
    g_vis.add_argument(
        "--plot-every-k",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Stride hint when subsampling; also used when max-timeline-columns is 0 "
            "for the legacy timeline (must divide step count). Default: 1."
        ),
    )
    g_vis.add_argument(
        "--brains-per-page",
        type=int,
        choices=(2, 4),
        default=4,
        metavar="N",
        help="Brains per page in the PDF report. Default: 4.",
    )
    g_vis.add_argument(
        "--include-timeline",
        action="store_true",
        help="Also write the wide horizontal timeline (*_brain_timeline.pdf/png per --format).",
    )

    g_extra = p.add_argument_group("Extra outputs")
    g_extra.add_argument(
        "--save-preds",
        action="store_true",
        help="Save raw predictions to *_preds.npz.",
    )
    g_extra.add_argument(
        "--brain-mp4",
        action="store_true",
        help="Export brain-only MP4 (requires ffmpeg on PATH).",
    )

    return p


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_arg_parser().parse_args()


def main() -> int:
    """Load TRIBE, run predict on the video, then export the paginated report and optional extras."""
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_folder = args.cache_folder.expanduser().resolve()
    cache_folder.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"PyTorch device: {device}")

    ext_update = extractor_config_update()
    if ext_update:
        print(
            "Using CPU for text/audio/video feature extractors (no CUDA; "
            "avoids hub cuda config on Mac).",
        )

    model = TribeModel.from_pretrained(
        args.repo,
        cache_folder=str(cache_folder),
        device=device,
        config_update=ext_update or None,
    )

    if args.audio_only_events:
        print("Building events without WhisperX transcription…")
        events = events_from_video(video_path, audio_only=True)
    else:
        if not shutil.which("uvx"):
            print(
                "Error: `uvx` not found. TRIBE calls `uvx whisperx` for transcription.\n"
                "  • In your venv: pip install uv\n"
                "  • Or: https://docs.astral.sh/uv/getting-started/installation/\n"
                "  • Or: --audio-only-events (no word-level text).",
                file=sys.stderr,
            )
            return 1
        apply_whisper_cpu_float32_patch()
        print("Building events (WhisperX via uvx, segmentation, …)…")
        events = events_from_video(video_path, audio_only=False)

    print("Running inference (may take a while; cache speeds up repeats)…")
    try:
        preds, segments = model.predict(events, verbose=True)
    except Exception as e:
        if is_hf_gated_llama_access_error(e):
            print_llama_gated_help()
            return 1
        raise
    print(f"Predictions shape: {preds.shape} (time × vertices)")

    preds_plot, segments_plot, plot_pe_k, timeline_msg = prepare_timeline_for_plot(
        preds,
        segments,
        max_columns=args.max_timeline_columns,
        plot_every_k=args.plot_every_k,
    )
    print(timeline_msg)

    stem = video_path.stem
    if args.save_preds:
        pred_path = out_dir / f"{stem}_preds.npz"
        np.savez_compressed(
            pred_path,
            preds=preds,
            video_path=str(video_path),
            device=device,
        )
        print(f"Saved predictions: {pred_path}")

    views = parse_views(args.views)
    plot_kwargs: dict = {"views": views, "cmap": "hot"}
    if args.norm_percentile is not None:
        plot_kwargs["norm_percentile"] = args.norm_percentile

    plotter = PlotBrain()
    show_stimuli = not args.no_stimuli
    wanted = {x.strip().lower()
              for x in args.out_formats.split(",") if x.strip()}

    report_pdf = out_dir / \
        f"{stem}_brain_report.pdf" if "pdf" in wanted else None
    report_png_p1 = (
        out_dir / f"{stem}_brain_report_page01.png"
        if "png" in wanted and not args.include_timeline
        else None
    )

    if report_pdf is not None or report_png_p1 is not None:
        print("Rendering paginated PDF report (PyVista off-screen)…")
        export_brain_report(
            plotter,
            preds_plot,
            segments_plot,
            pdf_path=report_pdf,
            png_first_page_path=report_png_p1,
            video_name=video_path.name,
            brains_per_page=args.brains_per_page,
            views=views,
            cmap="hot",
            norm_percentile=args.norm_percentile,
            show_stimuli=show_stimuli,
            figure_dpi=args.figure_dpi,
        )
        if report_pdf is not None:
            print(f"Report PDF: {report_pdf}")
        if report_png_p1 is not None and report_png_p1.is_file():
            print(f"Report preview PNG (page 1): {report_png_p1}")

    if args.include_timeline:
        print("Rendering horizontal timeline (--include-timeline)…")
        try:
            fig = plotter.plot_timesteps(
                preds_plot,
                segments=segments_plot,
                show_stimuli=show_stimuli,
                plot_every_k_timesteps=plot_pe_k,
                **plot_kwargs,
            )
        except AttributeError as e:
            if show_stimuli and "get_frame" in str(e):
                print(
                    "Warning: missing video clip for at least one segment; "
                    "redrawing timeline without video/audio/word overlays.",
                    file=sys.stderr,
                )
                fig = plotter.plot_timesteps(
                    preds_plot,
                    segments=segments_plot,
                    show_stimuli=False,
                    plot_every_k_timesteps=plot_pe_k,
                    **plot_kwargs,
                )
            else:
                raise
        fig.suptitle(f"TRIBE v2 — {video_path.name}", fontsize=12, y=1.02)
        for ext in ("pdf", "png"):
            if ext not in wanted:
                continue
            out_path = out_dir / f"{stem}_brain_timeline.{ext}"
            fig.savefig(out_path, bbox_inches="tight", dpi=args.figure_dpi)
            print(f"Timeline: {out_path}")
        plt.close(fig)

    if args.brain_mp4:
        mp4_path = out_dir / f"{stem}_brain_only.mp4"
        print(f"Exporting brain MP4 (ffmpeg): {mp4_path} …")
        plotter.plot_timesteps_mp4(
            preds_plot,
            filepath=str(mp4_path),
            segments=segments_plot,
            **plot_kwargs,
        )
        print(f"Brain MP4: {mp4_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
