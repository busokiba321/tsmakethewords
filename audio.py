from __future__ import annotations
import math
import re
from collections import Counter
from pathlib import Path


def has_words(text: str) -> bool:
	return bool(re.search(r"\w", (text or "").strip()))


def transcript_stats(text: str):
	t = (text or "").strip()
	words = t.split()
	return {"chars": len(t), "words": len(words), "head": t[:120], "tail": t[-120:] if len(t) > 120 else t}


def words_per_minute(word_count: int, duration_s: float) -> float:
	if duration_s <= 0:
		return 0.0
	return float(word_count) / (float(duration_s) / 60.0)


def repetition_ratio(text: str) -> float:
	ws = (text or "").strip().lower().split()
	if not ws:
		return 0.0
	c = Counter(ws)
	return max(c.values()) / len(ws)


def is_usable_text_file(p: Path, min_chars: int = 20) -> bool:
	if not p.exists():
		return False
	try:
		txt = p.read_text(encoding="utf-8", errors="ignore").strip()
	except Exception:
		return False
	return len(txt) >= min_chars


def combine_files(out_path: Path, parts: list[Path]):
	out_path.parent.mkdir(parents=True, exist_ok=True)
	texts: list[str] = []
	for p in parts:
		if not p.exists():
			continue
		try:
			texts.append(p.read_text(encoding="utf-8", errors="ignore").rstrip()  "\n")
		except Exception:
			continue
	out_path.write_text("".join(texts), encoding="utf-8")


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
	import wave, webrtcvad
	vad = webrtcvad.Vad(aggressiveness)
	with wave.open(str(wav_path), "rb") as wf:
		sr = wf.getframerate()
		nchan = wf.getnchannels()
		sampwidth = wf.getsampwidth()
		assert nchan == 1
		assert sampwidth == 2
		speech_frames = 0
		total_frames = 0
		while True:
			frame = wf.readframes(int(sr * (frame_ms / 1000.0)))
			if not frame:
				break
			total_frames += 1
			if vad.is_speech(frame, sr):
				speech_frames += 1
	ratio = (speech_frames / total_frames) if total_frames else 0.0
	return {"speech_frames": speech_frames, "total_frames": total_frames, "ratio": ratio, "frame_ms": frame_ms}


def vad_likely_speech(wav_path: Path, *, min_speech_ms: int = 900, min_ratio: float = 0.05) -> bool:
	st = vad_stats(wav_path)
	speech_ms = st["speech_frames"] * st["frame_ms"]
	return speech_ms >= min_speech_ms and st["ratio"] >= min_ratio


def extract_segment_wav(src_wav: Path, dst_wav: Path, start_s: float, dur_s: float = 15.0) -> None:
	import ffmpeg
	(
		ffmpeg.input(str(src_wav), ss=max(0.0, float(start_s)))
		.output(str(dst_wav), t=float(dur_s), ar=16000, ac=1, acodec="pcm_s16le", format="wav", loglevel="error")
		.overwrite_output()
		.run()
	)


def nemo_probe_has_text(model, wav_path: Path, tmp_dir: Path, *, probe_dur_s: float = 15.0) -> bool:
	dur = float(wav_info(wav_path).get("duration_s", 0.0) or 0.0)
	if dur <= 0.0:
		return False
	starts = [0.0]
	if dur > probe_dur_s * 2:
		starts.append(max(0.0, dur / 2.0 - probe_dur_s / 2.0))
	if dur > probe_dur_s * 3:
		starts.append(max(0.0, dur - probe_dur_s))
	tmp_dir.mkdir(parents=True, exist_ok=True)
	for i, st in enumerate(starts):
		seg = tmp_dir / f"probe_{wav_path.stem}_{i}.wav"
		try:
			extract_segment_wav(wav_path, seg, st, probe_dur_s)
			out = model.transcribe([str(seg)], timestamps=False)
			if out:
				text = (getattr(out[0], "text", "") or "").strip()
				if text:
					return True
		except Exception:
			continue
		finally:
			try:
				if seg.exists():
					seg.unlink()
			except Exception:
				pass
	return False


def split_wav_fixed(in_wav: Path, out_dir: Path, segment_s: int = 30):
	import ffmpeg
	out_dir.mkdir(parents=True, exist_ok=True)
	out_pattern = str(out_dir / "out%03d.wav")
	(
		ffmpeg.input(str(in_wav))
		.output(out_pattern, f="segment", segment_time=segment_s, reset_timestamps=1, ar=16000, ac=1, acodec="pcm_s16le", loglevel="error")
		.overwrite_output()
		.run()
	)
	return sorted(out_dir.glob("out*.wav"))


def convert_and_split(in_path: Path, tmp_dir: Path, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs", max_duration_s: int = 300):
	import ffmpeg
	tmp_dir.mkdir(parents=True, exist_ok=True)
	base = in_path.stem
	wav_path = tmp_dir / f"{base}.wav"
	(
		ffmpeg.input(str(in_path))
		.output(str(wav_path), ar=16000, ac=1, acodec="pcm_s16le", format="wav", loglevel="error")
		.overwrite_output()
		.run()
	)
	out_dir = tmp_dir / f"{base}_chunks"
	out_dir.mkdir(parents=True, exist_ok=True)
	out_pattern = str(out_dir / f"{base}_%03d.wav")
	(
		ffmpeg.input(str(wav_path))
		.output(out_pattern, f="segment", segment_time=max_duration_s, reset_timestamps=1, ar=16000, ac=1, acodec="pcm_s16le", loglevel="error")
		.overwrite_output()
		.run()
	)
	return base, sorted(out_dir.glob(f"{base}_*.wav"))