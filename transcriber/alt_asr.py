from __future__ import annotations

from pathlib import Path

from .config import AppConfig

_ALT_MODEL = None
_ALT_MODEL_KEY: tuple[str, str, str, bool] | None = None
_ALT_MODEL_DEVICE: str | None = None
_ALT_MODEL_COMPUTE: str | None = None
_ALT_MODEL_ID: str | None = None
_ALT_MODEL_ALLOW_DOWNLOAD: bool | None = None


def alt_asr_runtime_info() -> dict[str, object]:
    return {
        "backend": "openai-whisper/torch",
        "model": _ALT_MODEL_ID or "",
        "device": _ALT_MODEL_DEVICE or "",
        "compute_type": _ALT_MODEL_COMPUTE or "",
        "allow_download": bool(_ALT_MODEL_ALLOW_DOWNLOAD) if _ALT_MODEL_ALLOW_DOWNLOAD is not None else False,
    }


def run_alt_asr_on_chunk(cfg: AppConfig, wav_path: Path) -> str:
    """Transcribe a single WAV chunk with OpenAI Whisper (PyTorch) as alt ASR (GPU-only)."""
    try:
        import whisper  # openai-whisper
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai-whisper is not installed; cannot run alt ASR") from e

    global _ALT_MODEL, _ALT_MODEL_KEY, _ALT_MODEL_COMPUTE, _ALT_MODEL_DEVICE, _ALT_MODEL_ID, _ALT_MODEL_ALLOW_DOWNLOAD

    allow_download = bool(getattr(cfg, "alt_asr_allow_download", False))

    model_id = str(cfg.alt_asr_model)
    # Force GPU-only by design (no CPU fallback). If CUDA isn't available, fail fast.
    if not torch.cuda.is_available():
        raise RuntimeError("alt ASR requires CUDA but torch.cuda.is_available() is False")
    device = "cuda"
    # openai-whisper uses fp16 on CUDA by default; expose as metadata only.
    preferred_compute = "fp16"


    # Cache the loaded model to avoid reloading per chunk.
    key = (model_id, device, preferred_compute, (not allow_download))
    if _ALT_MODEL is None or _ALT_MODEL_KEY != key:
        try:
            _ALT_MODEL = whisper.load_model(model_id, device="cuda")
        except Exception as e:
            raise RuntimeError(f"alt ASR model load failed. model='{model_id}' err='{e}'") from e
        _ALT_MODEL_KEY = key
        _ALT_MODEL_COMPUTE = preferred_compute
        _ALT_MODEL_DEVICE = device
        _ALT_MODEL_ID = model_id
        _ALT_MODEL_ALLOW_DOWNLOAD = allow_download

    model = _ALT_MODEL
 
    try:
        res = model.transcribe(
            str(wav_path),
            language=None,
            task="transcribe",
            fp16=True,
            temperature=0.0,
            condition_on_previous_text=False,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"alt ASR transcription failed for chunk: {wav_path} err='{e}'") from e

    txt = (res.get("text", "") or "").strip()
    return txt
