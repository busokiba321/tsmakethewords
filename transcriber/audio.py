from __future__ import annotations

import math
import subprocess
import sys
import uuid
from pathlib import Path

MAX_DURATION = 5 * 60


def extract_segment_wav(src_wav: Path, dst_wav: Path, start_s: float, dur_s: float = 15.0) -> None:
    import ffmpeg

    (
        ffmpeg.input(str(src_wav), ss=max(0.0, float(start_s)))
        .output(
            str(dst_wav),
            t=float(dur_s),
            ar=16000,
            ac=1,
            acodec="pcm_s16le",
            format="wav",
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )


def wav_info(path: Path):
    import soundfile as sf

    i = sf.info(str(path))
    dur = 0.0 if i.samplerate == 0 else (i.frames / float(i.samplerate))
    return {"samplerate": i.samplerate, "channels": i.channels, "frames": i.frames, "duration_s": dur}


def wav_rms(path: Path, max_seconds: float = 6.0):
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    n = min(len(data), int(sr * max_seconds))
    if n <= 0:
        return 0.0
    x = data[:n]
    s = float((x * x).mean())
    return math.sqrt(s) if s > 0 else 0.0


def vad_stats(wav_path: Path, aggressiveness: int = 3, frame_ms: int = 30):
    import wave
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            return {"total_frames": 0, "speech_frames": 0, "speech_ratio": 0.0, "speech_ms": 0}
        pcm = wf.readframes(wf.getnframes())

    frame_len = int(16000 * (frame_ms / 1000.0)) * 2
    if frame_len <= 0:
        return {"total_frames": 0, "speech_frames": 0, "speech_ratio": 0.0, "speech_ms": 0}

    total = 0
    speech = 0
    for i in range(0, len(pcm) - frame_len + 1, frame_len):
        frame = pcm[i : i + frame_len]
        total += 1
        if vad.is_speech(frame, 16000):
            speech += 1

    if total == 0:
        return {"total_frames": 0, "speech_frames": 0, "speech_ratio": 0.0, "speech_ms": 0}
    ratio = speech / total
    return {"total_frames": total, "speech_frames": speech, "speech_ratio": ratio, "speech_ms": int(speech * frame_ms)}


def vad_likely_speech(wav_path: Path, *, min_speech_ms: int = 900, min_ratio: float = 0.05) -> bool:
    st = vad_stats(wav_path, aggressiveness=3, frame_ms=30)
    return (st["speech_ms"] >= min_speech_ms) and (st["speech_ratio"] >= min_ratio)


def nemo_probe_has_text(model, wav_path: Path, tmp_dir: Path, *, probe_dur_s: float = 15.0) -> bool:
    from .utils import has_words

    meta = wav_info(wav_path)
    dur = float(meta.get("duration_s", 0.0) or 0.0)
    if dur <= 0.5:
        return False

    points = [5.0, max(0.0, dur * 0.5 - probe_dur_s * 0.5), max(0.0, dur - probe_dur_s - 1.0)]
    probe_root = tmp_dir / "_nemo_probe"
    probe_root.mkdir(parents=True, exist_ok=True)

    for i, t0 in enumerate(points):
        out_wav = probe_root / f"{wav_path.stem}_probe{i}.wav"
        try:
            extract_segment_wav(wav_path, out_wav, t0, dur_s=probe_dur_s)
        except Exception:
            continue

        try:
            try:
                out = model.transcribe([str(out_wav)], timestamps=False, verbose=False)
            except TypeError:
                out = model.transcribe([str(out_wav)], timestamps=False)
        except Exception:
            continue

        if out:
            txt = (getattr(out[0], "text", "") or "").strip()
            if has_words(txt):
                return True
    return False


def enhance_speech_wav(in_wav: Path, out_wav: Path) -> None:
    import ffmpeg

    af = "highpass=f=80,lowpass=f=7800,afftdn,dynaudnorm"
    (
        ffmpeg.input(str(in_wav))
        .output(str(out_wav), ar=16000, ac=1, format="wav", acodec="pcm_s16le", af=af, loglevel="error")
        .overwrite_output()
        .run()
    )


def demucs_extract_vocals(in_wav: Path, out_wav: Path, model_name: str = "htdemucs") -> None:
    tmp_root = out_wav.parent / "_demucs"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model_name,
        "--two-stems=vocals",
        "-o",
        str(tmp_root),
        str(in_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs failed. stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}")

    voc = tmp_root / model_name / in_wav.stem / "vocals.wav"
    if not voc.exists():
        raise RuntimeError(f"Demucs completed but vocals.wav not found at: {voc}")

    import ffmpeg

    (
        ffmpeg.input(str(voc))
        .output(str(out_wav), ar=16000, ac=1, acodec="pcm_s16le", format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )


def probe(path: Path):
    import ffmpeg

    try:
        meta = ffmpeg.probe(str(path))
    except ffmpeg.Error as e:
        msg = e.stderr.decode("utf-8", errors="ignore").strip() if getattr(e, "stderr", None) else "ffprobe failed"
        raise RuntimeError(f"ffprobe could not read '{path}'. Details: {msg}")

    stream = next((s for s in meta["streams"] if s.get("codec_type") == "audio"), meta["streams"][0])
    fmt = meta.get("format", {})
    raw_dur = fmt.get("duration") or stream.get("duration")
    return {
        "format": path.suffix.lower(),
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration": float(raw_dur) if raw_dur is not None else 0.0,
    }


def convert_and_split(
    in_path: Path,
    tmp_dir: Path,
    *,
    enhance_speech: bool = False,
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    max_duration_s: int = MAX_DURATION,
):
    import ffmpeg
    import soundfile as sf

    base = tmp_dir / uuid.uuid4().hex
    conv = base.with_suffix(".wav")

    (
        ffmpeg.input(str(in_path))
        .output(str(conv), ar=16000, ac=1, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )

    if enhance_speech:
        enh = base.with_name(base.name + "_enh.wav")
        enhance_speech_wav(conv, enh)
        conv.unlink(missing_ok=True)
        conv = enh

    if use_demucs:
        voc = base.with_name(base.name + "_vocals.wav")
        demucs_extract_vocals(conv, voc, model_name=demucs_model)
        conv.unlink(missing_ok=True)
        conv = voc

    info = sf.info(str(conv))
    if info.duration <= max_duration_s:
        return [conv]

    pattern = str(base) + "_%03d.wav"
    (
        ffmpeg.input(str(conv))
        .output(pattern, ar=16000, ac=1, f="segment", segment_time=max_duration_s, reset_timestamps=1, loglevel="error")
        .overwrite_output()
        .run()
    )
    conv.unlink(missing_ok=True)
    return sorted(tmp_dir.glob(base.name + "_*.wav"))


def split_wav_fixed(in_wav: Path, out_dir: Path, segment_s: int = 30):
    import ffmpeg

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = in_wav.stem
    pattern = str(out_dir / f"{stem}_%03d.wav")
    (
        ffmpeg.input(str(in_wav))
        .output(
            pattern,
            format="segment",
            segment_time=segment_s,
            reset_timestamps=1,
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )
    return sorted(out_dir.glob(f"{stem}_*.wav"))
