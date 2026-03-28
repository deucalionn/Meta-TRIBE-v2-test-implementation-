"""TRIBE helpers: device, hub overrides, Whisper patch, events, timeline subsampling."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralset.segments import Segment

import tribev2.eventstransforms as et
from tribev2.demo_utils import get_audio_and_text_events


def resolve_device(choice: str) -> str:
    """Resolve CLI device choice to a PyTorch device string (cuda, mps, or cpu)."""
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extractor_config_update() -> dict:
    """Hub config forcing feature extractors to CPU when CUDA is unavailable.

    Avoids hub configs that require CUDA on Macs with only MPS; the TRIBE core can still use MPS.
    """
    if torch.cuda.is_available():
        return {}
    return {
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.device": "cpu",
    }


def parse_views(s: str) -> str | list[str]:
    """Parse comma-separated cortical view names into one string or a list."""
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) == 1:
        return parts[0]
    return parts


def events_from_video(video_path: Path, *, audio_only: bool) -> pd.DataFrame:
    """Build TRIBE event table for one video; ``audio_only`` skips Whisper word events."""
    event = {
        "type": "Video",
        "filepath": str(video_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    return get_audio_and_text_events(pd.DataFrame([event]), audio_only=audio_only)


def apply_whisper_cpu_float32_patch() -> None:
    """Patch TRIBE so WhisperX uses float32 on CPU (float16 fails in ctranslate2)."""
    log = logging.getLogger("tribev2.eventstransforms")

    def _get_transcript_from_audio(
        wav_filename: Path,
        language: str,
    ) -> pd.DataFrame:
        language_codes = dict(
            english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
        )
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        with tempfile.TemporaryDirectory() as output_dir:
            log.info(
                "Running whisperx via uvx (compute_type=%s, device=%s)...",
                compute_type,
                device,
            )
            align = (
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else ""
            )
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "16",
                "--align_model",
                align,
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]
            env = {
                k: v for k, v in os.environ.items() if k != "MPLBACKEND"
            }
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words = []
        for i, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                words.append(
                    {
                        "text": word["word"].replace('"', ""),
                        "start": word["start"],
                        "duration": word["end"] - word["start"],
                        "sequence_id": i,
                        "sentence": sentence,
                    }
                )

        return pd.DataFrame(words)

    et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        _get_transcript_from_audio
    )


def is_hf_gated_llama_access_error(exc: BaseException) -> bool:
    """True if the error likely indicates missing Hugging Face access to gated Llama."""
    s = str(exc).lower()
    return (
        "gated repo" in s
        or "not in the authorized list" in s
        or ("403" in s and "meta-llama" in s)
    )


def print_llama_gated_help() -> None:
    """Print instructions to stderr for unlocking gated Llama on Hugging Face."""
    print(
        "\nAccess denied to gated Llama 3.2 (TRIBE text encoder).\n"
        "  1. Open https://huggingface.co/meta-llama/Llama-3.2-3B\n"
        "  2. Log in and accept the license / request access from Meta\n"
        "  3. Wait until access is granted\n"
        "  4. Ensure `hf auth login` uses the same HF account (`hf auth whoami`)\n",
        file=sys.stderr,
    )


def stride_for_timesteps(n_timesteps: int, requested: int) -> int:
    """Pick stride K so ``n_timesteps`` divides evenly (TRIBE timeline requirement)."""
    k = max(1, requested)
    if n_timesteps % k == 0:
        return k
    for cand in range(min(k, n_timesteps), 0, -1):
        if n_timesteps % cand == 0:
            print(
                f"Note: --plot-every-k={requested} is incompatible with {n_timesteps} "
                f"steps; using stride {cand} instead.",
                file=sys.stderr,
            )
            return cand
    return 1


def prepare_timeline_for_plot(
    preds: np.ndarray,
    segments: list[Segment],
    *,
    max_columns: int,
    plot_every_k: int,
) -> tuple[np.ndarray, list[Segment], int, str]:
    """Subsample predictions and segments for display without re-running the model.

    If ``max_columns`` > 0, use evenly spaced indices plus the last step. If 0, keep all steps
    and use ``stride_for_timesteps`` for the legacy wide timeline. Returns subsampled preds and
    segments, stride for ``plot_timesteps``, and a short log line.
    """
    n = int(preds.shape[0])
    if n <= 0:
        return preds, segments, 1, "Timeline: no steps"

    if max_columns <= 0:
        stride = stride_for_timesteps(n, plot_every_k)
        msg = (
            f"Timeline: {n} steps, showing 1 column every {stride} "
            f"(max-timeline-columns=0)"
        )
        return preds, segments, stride, msg

    step = max(1, int(np.ceil(n / max_columns)), max(1, plot_every_k))
    idx_list = np.arange(0, n, step, dtype=int).tolist()
    if not idx_list:
        idx_list = [0]
    if idx_list[-1] != n - 1:
        idx_list.append(n - 1)
    seen: set[int] = set()
    idx: list[int] = []
    for i in idx_list:
        if i not in seen:
            seen.add(i)
            idx.append(i)
    idx_a = np.array(idx, dtype=int)
    p = preds[idx_a]
    s = [segments[i] for i in idx]
    msg = (
        f"Display: {n} time steps -> {len(idx)} panels "
        f"(cap {max_columns})"
    )
    return p, s, 1, msg
