from __future__ import annotations

import math
import signal
import subprocess
import sys
import uuid
import shutil
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
            out_wav.unlink(missing_ok=True)
            continue

        if out:
            txt = (getattr(out[0], "text", "") or "").strip()
            if has_words(txt):
                out_wav.unlink(missing_ok=True)
                return True
        out_wav.unlink(missing_ok=True)
    # Best-effort: remove probe dir if empty
    try:
        probe_root.rmdir()
    except OSError:
        pass
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
    # Demucs is just preprocessing; if CUDA crashes (common on older GPUs / driver combos),
    # fallback to CPU so the pipeline still runs. Parakeet can remain on GPU.
    def _run_demucs(device: str) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            model_name,
            "--two-stems",
            "vocals",
            "-d",
            device,
            "-o",
            str(tmp_root),
            str(in_wav),
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    # Try CUDA first only if torch reports CUDA available; otherwise CPU directly.
    devices = ["cpu"]
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            devices = ["cuda", "cpu"]
    except Exception:
        # If torch isn't importable here for any reason, don't block the pipeline.
        devices = ["cpu"]

    proc: subprocess.CompletedProcess | None = None
    last_err: str | None = None
    for dev in devices:
        proc = _run_demucs(dev)
        if proc.returncode == 0:
            break

        # Keep a useful error summary for the final exception
        rc = proc.returncode
        sig = -rc if rc < 0 else 0
        sig_name = signal.Signals(sig).name if sig else ""
        out_preview = (proc.stdout or proc.stderr or "").strip().replace("\n", " ")[:400]
        last_err = f"device={dev} returncode={rc}" + (f" signal={sig_name}" if sig_name else "") + (f" preview={out_preview}" if out_preview else "")
    else:
        # Fell through without success
        assert proc is not None
        raise RuntimeError(
            "Demucs failed.\n"
            f"  in_wav: {in_wav}\n"
            f"  model: {model_name}\n"
            f"  last_attempt: {last_err}\n"
            f"  stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    voc = tmp_root / model_name / in_wav.stem / "vocals.wav"
    if not voc.exists():
        raise RuntimeError(f"Demucs completed but vocals.wav not found at: {voc}")

    import ffmpeg

    try:
        (
            ffmpeg.input(str(voc))
            .output(str(out_wav), ar=16000, ac=1, acodec="pcm_s16le", format="wav", loglevel="error")
            .overwrite_output()
            .run()
        )
    finally:
        # Always clean Demucs outputs (can be huge)
        shutil.rmtree(tmp_root, ignore_errors=True)

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
    conv = base.with_suffix(".wav")  # initial 16k mono wav

    (
        ffmpeg.input(str(in_path))
        .output(str(conv), ar=16000, ac=1, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )

    # IMPORTANT: For long files, DO NOT run demucs on the full-length wav.
    # Split first, then run enhance/demucs per chunk to avoid OOM/SIGKILL.

    info = sf.info(str(conv))
    if info.duration <= max_duration_s:
        chunks = [conv]
        conv_is_chunk = True
    else:
        pattern = str(base) + "_%03d.wav"
        (
            ffmpeg.input(str(conv))
            .output(
                pattern,
                ar=16000,
                ac=1,
                f="segment",
                segment_time=max_duration_s,
                reset_timestamps=1,
                loglevel="error",
            )
            .overwrite_output()
            .run()
        )
        chunks = sorted(tmp_dir.glob(base.name + "_*.wav"))
        conv_is_chunk = False

    # We can delete the full-length wav once we’ve created chunk files.
    # If the full wav itself is the only chunk, keep it for processing below.
    if not conv_is_chunk:
        conv.unlink(missing_ok=True)

    final_chunks: list[Path] = []
    for i, ch in enumerate(chunks):
        cur = ch

        if enhance_speech:
            enh = cur.with_name(cur.stem + "_enh.wav")
            enhance_speech_wav(cur, enh)
            # remove prior stage
            if cur != enh:
                cur.unlink(missing_ok=True)
            cur = enh

        if use_demucs:
            voc = cur.with_name(cur.stem + "_vocals.wav")
            demucs_extract_vocals(cur, voc, model_name=demucs_model)
            # remove prior stage
            if cur != voc:
                cur.unlink(missing_ok=True)
            cur = voc

        final_chunks.append(cur)

    # If the only chunk was the original conv wav and we produced a new file,
    # make sure the original is removed.
    if conv_is_chunk and conv not in final_chunks:
        conv.unlink(missing_ok=True)

    return final_chunks


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
