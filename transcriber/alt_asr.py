from __future__ import annotations

from pathlib import Path

from .config import AppConfig


def run_alt_asr_on_chunk(cfg: AppConfig, wav_path: Path) -> str:
    """Transcribe a single WAV chunk with Faster-Whisper (CT2) alt model."""
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover - import error depends on env
        raise RuntimeError("faster-whisper is not installed; cannot run alt ASR") from e

    model_kwargs = {
        "device": cfg.alt_asr_device,
        "compute_type": cfg.alt_asr_compute_type,
        "local_files_only": True,
    }

    try:
        model = WhisperModel(cfg.alt_asr_model, **model_kwargs)
    except TypeError:
        # Older faster-whisper releases may not expose local_files_only.
        model_kwargs.pop("local_files_only", None)
        model = WhisperModel(cfg.alt_asr_model, **model_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"failed to load alt ASR model '{cfg.alt_asr_model}' from local cache"
        ) from e

    try:
        segments, _ = model.transcribe(
            str(wav_path),
            beam_size=int(cfg.alt_asr_beam_size),
            temperature=0.0,
            vad_filter=bool(cfg.alt_asr_vad_filter),
            condition_on_previous_text=False,
            word_timestamps=False,
        )
    except Exception as e:
        raise RuntimeError(f"alt ASR transcription failed for chunk: {wav_path}") from e

    text_parts = []
    for seg in segments:
        t = (getattr(seg, "text", "") or "").strip()
        if t:
            text_parts.append(t)

    return " ".join(text_parts).strip()
