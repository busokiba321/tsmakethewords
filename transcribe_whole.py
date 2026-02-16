#!/home/archlinux/Documents/Projects/Transcriber/parakeet-env/bin/python
import argparse, os, shutil, uuid, json, math
import sys, time, re, subprocess
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# maximum chunk length in seconds (5 min = 300 s)
MAX_DURATION = 5 * 60
 
CONFIG_PATH = Path("transcriber.toml")

@dataclass
class AppConfig:
	audio_dir: str = "audio"
	default_timestamps: bool = False
	show_progress: bool = True
	nemo_retries: int = 5
	enhance_speech: bool = False
	use_demucs: bool = False
	demucs_model: str = "htdemucs"
	adjudicate: bool = True
	adjudicate_only_when_flagged: bool = True
	llama_bin: str = "llama-cli"  # or "llama" / "main" depending on build
	llama_model: str = "models/mistral-7b-instruct.gguf"
	llama_ctx: int = 4096
	llama_gpu_layers: int = 999
	llama_threads: int = 6
	llama_seed: int = 42
	llama_temp: float = 0.0
	llama_top_p: float = 1.0
	llama_max_tokens: int = 900
	llama_grammar: str = ""  # optional path to GBNF grammar file for llama.cpp

def load_config() -> AppConfig:
	"""Load config from transcriber.toml (tiny TOML, no third-party deps)."""
	if not CONFIG_PATH.exists():
		return AppConfig()
	import tomllib

	data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

	# Ignore any legacy keys from older designs.
	return AppConfig(
		audio_dir=str(data.get("audio_dir", "audio")),
		default_timestamps=bool(data.get("default_timestamps", False)),
		show_progress=bool(data.get("show_progress", True)),
		nemo_retries=int(data.get("nemo_retries", 5)),
		enhance_speech=bool(data.get("enhance_speech", False)),
		use_demucs=bool(data.get("use_demucs", False)),
		demucs_model=str(data.get("demucs_model", "htdemucs")),
		adjudicate=bool(data.get("adjudicate", True)),
		adjudicate_only_when_flagged=bool(data.get("adjudicate_only_when_flagged", True)),
		llama_bin=str(data.get("llama_bin", "llama-cli")),
		llama_model=str(data.get("llama_model", "models/mistral-7b-instruct.gguf")),
		llama_ctx=int(data.get("llama_ctx", 4096)),
		llama_gpu_layers=int(data.get("llama_gpu_layers", 999)),
		llama_threads=int(data.get("llama_threads", 6)),
		llama_seed=int(data.get("llama_seed", 42)),
		llama_temp=float(data.get("llama_temp", 0.0)),
		llama_top_p=float(data.get("llama_top_p", 1.0)),
		llama_max_tokens=int(data.get("llama_max_tokens", 900)),
		llama_grammar=str(data.get("llama_grammar", "")),
	)

def save_config(cfg: AppConfig) -> None:
	"""Write transcriber.toml manually (tiny file, avoids extra deps)."""
	text = (
		f'audio_dir = "{cfg.audio_dir}"\n'
		f"default_timestamps = {str(cfg.default_timestamps).lower()}\n"
		f"show_progress = {str(cfg.show_progress).lower()}\n"
		f"nemo_retries = {int(cfg.nemo_retries)}\n"
		f"enhance_speech = {str(cfg.enhance_speech).lower()}\n"
		f"use_demucs = {str(cfg.use_demucs).lower()}\n"
		f'demucs_model = "{cfg.demucs_model}"\n'
		f"adjudicate = {str(cfg.adjudicate).lower()}\n"
		f"adjudicate_only_when_flagged = {str(cfg.adjudicate_only_when_flagged).lower()}\n"
		f'llama_bin = "{cfg.llama_bin}"\n'
		f'llama_model = "{cfg.llama_model}"\n'
		f"llama_ctx = {int(cfg.llama_ctx)}\n"
		f"llama_gpu_layers = {int(cfg.llama_gpu_layers)}\n"
		f"llama_threads = {int(cfg.llama_threads)}\n"
		f"llama_seed = {int(cfg.llama_seed)}\n"
		f"llama_temp = {float(cfg.llama_temp)}\n"
		f"llama_top_p = {float(cfg.llama_top_p)}\n"
		f"llama_max_tokens = {int(cfg.llama_max_tokens)}\n"
		f'llama_grammar = "{cfg.llama_grammar}"\n'
	)
	CONFIG_PATH.write_text(text, encoding="utf-8")


def _safe_execute(prompt):
	try:
		return prompt.execute()
	except KeyboardInterrupt:
		print("\n⛔ Closing (Ctrl+C).")
		raise SystemExit(130)

def _has_words(text: str) -> bool:
	return bool(re.search(r"\w", (text or "").strip()))

def _extract_segment_wav(src_wav: Path, dst_wav: Path, start_s: float, dur_s: float = 15.0) -> None:
	"""Extract a short WAV segment (16k mono PCM) for probing."""
	import ffmpeg
	(
		ffmpeg
		.input(str(src_wav), ss=max(0.0, float(start_s)))
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

def nemo_probe_has_text(model, wav_path: Path, tmp_dir: Path, *, probe_dur_s: float = 15.0) -> bool:
	"""
	Run a few short NeMo probes on the chunk using the SAME model.
	If any probe yields text, we treat the chunk as containing intermittent speech.
	This avoids VAD false positives on music.
	"""
	meta = wav_info(wav_path)
	dur = float(meta.get("duration_s", 0.0) or 0.0)
	if dur <= 0.5:
		return False

	# Probe positions: early, middle, late (clamped)
	points = [5.0, max(0.0, dur * 0.5 - probe_dur_s * 0.5), max(0.0, dur - probe_dur_s - 1.0)]
	probe_root = tmp_dir / "_nemo_probe"
	probe_root.mkdir(parents=True, exist_ok=True)

	for i, t0 in enumerate(points):
		out_wav = probe_root / f"{wav_path.stem}_probe{i}.wav"
		try:
			_extract_segment_wav(wav_path, out_wav, t0, dur_s=probe_dur_s)
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
			r = out[0]
			txt = (getattr(r, "text", "") or "").strip()
			if _has_words(txt):
				return True
	return False

def wav_info(path: Path):
	import soundfile as sf
	i = sf.info(str(path))
	dur = 0.0 if i.samplerate == 0 else (i.frames / float(i.samplerate))
	return {"samplerate": i.samplerate, "channels": i.channels, "frames": i.frames, "duration_s": dur}

def wav_rms(path: Path, max_seconds: float = 6.0):
	"""Approx RMS over first max_seconds to detect non-silent chunks."""
	import soundfile as sf
	data, sr = sf.read(str(path), dtype="float32", always_2d=True)
	n = min(len(data), int(sr * max_seconds))
	if n <= 0:
		return 0.0
	x = data[:n]
	s = float((x * x).mean())
	return math.sqrt(s) if s > 0 else 0.0

def vad_stats(wav_path: Path, aggressiveness: int = 3, frame_ms: int = 30):
	"""
	Return VAD stats for a WAV:
	- speech_frames / total_frames
	- speech_ms (approx)
	We use this for flagging only (not hard gates), because VAD can false-positive on music.
	Works on 16kHz mono PCM WAV (your pipeline already produces that).
	"""
	import webrtcvad
	import wave

	vad = webrtcvad.Vad(aggressiveness)
	with wave.open(str(wav_path), "rb") as wf:
		if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
			# Not in expected format; conservatively assume speech might exist.
			return {"total_frames": 0, "speech_frames": 0, "speech_ratio": 0.0, "speech_ms": 0}
		pcm = wf.readframes(wf.getnframes())

	frame_len = int(16000 * (frame_ms / 1000.0)) * 2  # bytes (16-bit)
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
	ratio = (speech / total)
	return {"total_frames": total, "speech_frames": speech, "speech_ratio": ratio, "speech_ms": int(speech * frame_ms)}

def vad_likely_speech(wav_path: Path, *, min_speech_ms: int = 900, min_ratio: float = 0.05) -> bool:
	"""
	Conservative speech-likelihood check.
	We require sustained speech activity to reduce false positives on music.
	"""
	st = vad_stats(wav_path, aggressiveness=3, frame_ms=30)
	return (st["speech_ms"] >= min_speech_ms) and (st["speech_ratio"] >= min_ratio)
 
def transcript_stats(text: str):
	t = (text or "").strip()
	words = t.split()
	return {
		"chars": len(t),
		"words": len(words),
		"head": t[:120],
		"tail": t[-120:] if len(t) > 120 else t,
	}

def _hypothesis_dir(base: str) -> Path:
	return Path("transcripts") / "hypotheses" / base

def _hypothesis_paths(base: str, idx: int) -> dict[str, Path]:
	"""
	Consistent per-chunk file layout for resumability and debugging.
	"""
	out_dir = _hypothesis_dir(base)
	stem = f"{base}_{idx:04d}"
	return {
		"dir": out_dir,
		"baseline": out_dir / f"{stem}.baseline.txt",
		"recover": out_dir / f"{stem}.recover.txt",
		"alt_asr": out_dir / f"{stem}.alt_asr.txt",  # future: distil-large-v3.5
		"adjudicated_json": out_dir / f"{stem}.adjudicated.json",
		"adjudicated_txt": out_dir / f"{stem}.adjudicated.txt",
	}

def _build_adjudication_prompt(hyps: dict[str, str], *, chunk_seconds: float, flags: list[str]) -> str:
	"""
	Extractive-only adjudication prompt:
	- Do NOT invent words not present in any hypothesis.
	- Prefer coherent phrasing, remove duplicates, fix corruption by choosing better hypothesis spans.
	- Output strict JSON only.
	"""
	def norm(s: str) -> str:
		return (s or "").strip()

	a = norm(hyps.get("baseline", ""))
	b = norm(hyps.get("recover", ""))
	c = norm(hyps.get("alt_asr", ""))

	flag_line = ", ".join(flags) if flags else "none"
	return (
		"You are an ASR transcript adjudicator.\n"
		"Task: produce the best possible transcript for this audio chunk by selecting and combining ONLY words\n"
		"that appear in the provided hypotheses. You MUST be extractive-only.\n"
		"\n"
		"Rules:\n"
		"1) Do NOT add or invent any words that do not appear in at least one hypothesis.\n"
		"2) Prefer coherent, grammatical phrasing.\n"
		"3) Remove duplicated phrases and obvious repetition.\n"
		"4) If uncertain between two options, choose the version that is present verbatim in one hypothesis.\n"
		"5) If all hypotheses are bad, return the least-wrong one; never hallucinate.\n"
		"\n"
		f"Chunk duration seconds: {chunk_seconds:.2f}\n"
		f"Flags: {flag_line}\n"
		"\n"
		"Return STRICT JSON only with keys:\n"
		'{"final": "...", "used": ["baseline","recover","alt_asr"], "confidence": "high|medium|low", "notes": ["tag",...]}\n'
		"\n"
		"Hypothesis baseline:\n"
		f"{a}\n"
		"\n"
		"Hypothesis recover:\n"
		f"{b}\n"
		"\n"
		"Hypothesis alt_asr:\n"
		f"{c}\n"
	)

def _run_llama_cpp_json(cfg: AppConfig, prompt: str) -> dict[str, Any] | None:
	"""
	Run llama.cpp as a subprocess and parse JSON output.
	We keep this dependency-free (no Python bindings).
	"""
	cmd = [
		cfg.llama_bin,
		"-m", cfg.llama_model,
		"-c", str(cfg.llama_ctx),
		"--seed", str(cfg.llama_seed),
		"--temp", str(cfg.llama_temp),
		"--top-p", str(cfg.llama_top_p),
		"-n", str(cfg.llama_max_tokens),
		"-t", str(cfg.llama_threads),
		"--gpu-layers", str(cfg.llama_gpu_layers),
		"-p", prompt,
	]
	if cfg.llama_grammar:
		cmd += ["--grammar-file", cfg.llama_grammar]

	try:
		proc = subprocess.run(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=180,
		)
	except Exception:
		return None

	if proc.returncode != 0:
		return None

	out = (proc.stdout or "").strip()
	if not out:
		return None

	# Attempt to locate first JSON object in output (llama.cpp sometimes emits extra tokens/newlines)
	m = re.search(r"\{.*\}", out, flags=re.DOTALL)
	if not m:
		return None
	try:
		return json.loads(m.group(0))
	except Exception:
		return None

def _choose_fallback_best(hyps: dict[str, str]) -> str:
	"""
	Safe fallback if adjudicator fails:
	Prefer the longest non-degenerate hypothesis.
	"""
	cands = []
	for k in ("recover", "alt_asr", "baseline"):
		t = (hyps.get(k) or "").strip()
		if not t:
			continue
		rep = _repetition_ratio(t)
		cands.append((rep, len(t.split()), k, t))
	if not cands:
		return ""
	# Prefer low repetition, then higher word count
	cands.sort(key=lambda x: (x[0], -x[1]))
	return cands[0][3]

def enhance_speech_wav(in_wav: Path, out_wav: Path) -> None:
	"""
	Lightweight ffmpeg filter chain to improve speech intelligibility in many cases.
	Not magic (won't remove music), but can help.
	"""
	import ffmpeg
	af = "highpass=f=80,lowpass=f=7800,afftdn,dynaudnorm"
	(
		ffmpeg
		.input(str(in_wav))
		.output(
			str(out_wav),
			ar=16000,
			ac=1,
			format="wav",
			acodec="pcm_s16le",
			af=af,
			loglevel="error",
		)
		.overwrite_output()
		.run()
	)


def demucs_extract_vocals(in_wav: Path, out_wav: Path, model_name: str = "htdemucs") -> None:
	"""
	Run Demucs and output a vocals-only WAV.
	Off by default. Requires `demucs` installed in the venv.
	We call the CLI to avoid heavy imports at startup.
	"""
	tmp_root = out_wav.parent / "_demucs"
	tmp_root.mkdir(parents=True, exist_ok=True)

	# Demucs CLI:
	# python -m demucs.separate -n htdemucs --two-stems=vocals -o <outdir> <input>
	cmd = [
		sys.executable, "-m", "demucs.separate",
		"-n", model_name,
		"--two-stems=vocals",
		"-o", str(tmp_root),
		str(in_wav),
	]
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if proc.returncode != 0:
		raise RuntimeError(
			"Demucs failed.\n"
			"Install with: pip install demucs\n"
			f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
		)

	# Expected output path:
	# <tmp_root>/<model_name>/<stem>/vocals.wav
	stem = in_wav.stem
	voc = tmp_root / model_name / stem / "vocals.wav"
	if not voc.exists():
		raise RuntimeError(f"Demucs completed but vocals.wav not found at: {voc}")

	# Convert to 16k mono PCM (Demucs often outputs 44.1k stereo)
	import ffmpeg
	(
		ffmpeg
		.input(str(voc))
		.output(str(out_wav), ar=16000, ac=1, acodec="pcm_s16le", format="wav", loglevel="error")
		.overwrite_output()
		.run()
	)

def probe(path: Path):
	"""Return dict with format, sample_rate, channels, duration."""
	import ffmpeg
	try:
		meta = ffmpeg.probe(str(path))
	except ffmpeg.Error as e:
		# Provide a friendly, actionable message.
		msg = (e.stderr.decode("utf-8", errors="ignore").strip()
			   if getattr(e, "stderr", None) else "ffprobe failed")
		raise RuntimeError(
			f"ffprobe could not read '{path}'.\n"
			f"Details: {msg}"
		)
	# pick the first audio stream (or default to index 0)
	stream = next((s for s in meta["streams"] if s.get("codec_type") == "audio"), meta["streams"][0])
	fmt = meta.get("format", {})
	# duration might live under format or stream
	raw_dur = fmt.get("duration") or stream.get("duration")
	return {
		"format": path.suffix.lower(),
		"sample_rate": int(stream["sample_rate"]),
		"channels": int(stream["channels"]),
		"duration": float(raw_dur) if raw_dur is not None else 0.0,
	}


def convert_and_split(in_path: Path, tmp_dir: Path, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs"):
	"""
	Convert any audio to 16 kHz mono WAV, then split into ≤ MAX_DURATION chunks.
	Returns a sorted list of chunk Paths.
	"""
	import ffmpeg
	import soundfile as sf

	base = tmp_dir / uuid.uuid4().hex
	conv = base.with_suffix(".wav")

	# 1) transcode
	(
		ffmpeg.input(str(in_path))
		.output(str(conv), ar=16000, ac=1, format="wav", loglevel="error")
		.overwrite_output()
		.run()
	)

	# Optional: speech enhancement
	if enhance_speech:
		enh = base.with_name(base.name + "_enh.wav")
		enhance_speech_wav(conv, enh)
		try:
			conv.unlink()
		except Exception:
			pass
		conv = enh

	# Optional: Demucs vocal separation (vocals-only)
	if use_demucs:
		voc = base.with_name(base.name + "_vocals.wav")
		demucs_extract_vocals(conv, voc, model_name=demucs_model)
		try:
			conv.unlink()
		except Exception:
			pass
		conv = voc

	# 2) split if needed
	info = sf.info(str(conv))
	if info.duration <= MAX_DURATION:
		return [conv]

	pattern = str(base) + "_%03d.wav"
	(
		ffmpeg.input(str(conv))
		.output(
			pattern,
			ar=16000,
			ac=1,
			f="segment",
			segment_time=MAX_DURATION,
			reset_timestamps=1,
			loglevel="error",
		)
		.overwrite_output()
		.run()
	)
	conv.unlink()
	return sorted(tmp_dir.glob(base.name + "_*.wav"))

def split_wav_fixed(in_wav: Path, out_dir: Path, segment_s: int = 30):
	"""Split a WAV into fixed-length segments (used for EOU realtime models)."""
	import ffmpeg
	out_dir.mkdir(parents=True, exist_ok=True)
	stem = in_wav.stem
	pattern = str(out_dir / f"{stem}_%03d.wav")
	(
		ffmpeg
		.input(str(in_wav))
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


def prepare_inputs(paths, tmp_dir, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs"):
	"""
	For each input file, probe and optionally convert+split.
	Returns a dict: {base_name: [chunk Paths...], ...}
	"""
	prepared = {}
	for p in paths:
		try:
			cfg = probe(p)
		except Exception as e:
			print(f"❌ Cannot process '{p}': {e}")
			print("   Tip: verify the file exists and ffmpeg/ffprobe can decode it.")
			raise SystemExit(2)
		needs = (
			cfg["format"] not in [".wav", ".flac"]
			or cfg["sample_rate"] != 16000
			or cfg["channels"] != 1
			or cfg["duration"] > MAX_DURATION
		)

		if needs:
			print(f"⚙️  Converting/splitting {p.name}")
			chunks = convert_and_split(p, tmp_dir, enhance_speech=enhance_speech, use_demucs=use_demucs, demucs_model=demucs_model)
		else:
			chunks = [p]

		base = p.stem
		prepared.setdefault(base, []).extend(chunks)

	return prepared

def prepare_inputs_with_progress(paths, tmp_dir, show_progress: bool = True, progress=None, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs"):
	"""
	Wrap prepare_inputs() with a Rich progress bar so conversion/splitting
	has visible progress (per file).
	"""
	if progress is None:
		progress_ctx, progress = _maybe_rich_progress(enabled=show_progress)
	else:
		progress_ctx = nullcontext()
	if progress is None:
		return prepare_inputs(paths, tmp_dir)

	total = len(paths)
	prepared = {}
	with progress_ctx:
		task = progress.add_task("Preparing audio files", total=total)
		for idx, p in enumerate(paths):
			progress.update(task, description=f"Preparing {Path(p).name} ({idx+1}/{total})")
			# This is still a blocking call, but user sees a spinner + status line.
			# (Optional heartbeat refresh below helps spinner feel alive.)
			prepared_one = prepare_inputs([Path(p)], tmp_dir, enhance_speech=enhance_speech, use_demucs=use_demucs, demucs_model=demucs_model)
			prepared.update(prepared_one)
			progress.advance(task, 1)
		progress.remove_task(task)
	return prepared

def _chunk_paths(base: str, idx: int):
	"""
	Paths for per-chunk and combined outputs.
	"""
	out_dir = Path("transcripts")
	out_dir.mkdir(exist_ok=True)

	idx_str = f"{idx:03d}"
	chunk_txt = out_dir / f"{base}_{idx_str}.txt"
	chunk_ts = out_dir / f"{base}_{idx_str}.timestamps.txt"

	combined_txt = out_dir / f"{base}.txt"
	combined_ts = out_dir / f"{base}.timestamps.txt"
	return chunk_txt, chunk_ts, combined_txt, combined_ts


def _combine_files(out_path: Path, parts: list[Path]):
	with open(out_path, "w", encoding="utf-8") as outf:
		for p in parts:
			if not p.exists():
				continue
			outf.write(p.read_text(encoding="utf-8"))
			outf.write("\n")

def _is_usable_text_file(p: Path, min_chars: int = 20) -> bool:
	"""
	Return True if file exists and contains meaningful text.
	Prevents resume logic from skipping empty/failed chunk outputs.
	"""
	if not p.exists():
		return False
	try:
		txt = p.read_text(encoding="utf-8", errors="ignore").strip()
	except Exception:
		return False
	return len(txt) >= min_chars


def _maybe_rich_progress(enabled: bool = True):
	"""
	Return (progress_ctx, progress_obj) where:
	- progress_ctx is a context manager
	- progress_obj is either a Rich Progress instance or None

	We keep this lazy-imported so Rich is optional in non-interactive CLI usage.
	"""
	if not enabled or not sys.stdout.isatty():
		return nullcontext(), None

	try:
		from rich.progress import (
			Progress,
			SpinnerColumn,
			TextColumn,
			BarColumn,
			MofNCompleteColumn,
			TimeElapsedColumn,
			TimeRemainingColumn,
		)
	except Exception:
		return nullcontext(), None

	progress = Progress(
		SpinnerColumn(),
		TextColumn("[progress.description]{task.description}"),
		BarColumn(),
		MofNCompleteColumn(),
		TimeElapsedColumn(),
		TimeRemainingColumn(),
	)
	return progress, progress


def _progress_desc(base: str, idx: int, total: int, mode: str) -> str:
	return f"{mode} {base}  chunk {idx+1}/{total}"

def wav_info(path: Path):
	import soundfile as sf
	i = sf.info(str(path))
	dur = 0.0 if i.samplerate == 0 else (i.frames / float(i.samplerate))
	return {"samplerate": i.samplerate, "channels": i.channels, "frames": i.frames, "duration_s": dur}

def wav_rms(path: Path, max_seconds: float = 6.0):
	"""Approx RMS over first max_seconds to detect non-silent chunks."""
	import soundfile as sf
	data, sr = sf.read(str(path), dtype="float32", always_2d=True)
	n = min(len(data), int(sr * max_seconds))
	if n <= 0:
		return 0.0
	x = data[:n]
	s = float((x * x).mean())
	return math.sqrt(s) if s > 0 else 0.0

def transcript_stats(text: str):
	t = (text or "").strip()
	words = t.split()
	return {
		"chars": len(t),
		"words": len(words),
		"head": t[:120],
		"tail": t[-120:] if len(t) > 120 else t,
	}

def words_per_minute(word_count: int, duration_s: float) -> float:
	if duration_s <= 0:
		return 0.0
	return float(word_count) / (float(duration_s) / 60.0)

def _repetition_ratio(text: str) -> float:
	"""Returns max single-token frequency / total tokens (0..1). High => repetitive / degenerate."""
	ws = (text or "").strip().lower().split()
	if not ws:
		return 0.0
	from collections import Counter
	c = Counter(ws)
	return max(c.values()) / len(ws)

def transcribe_chunks(base, chunks, model, timestamps, cfg: AppConfig, progress=None, chunk_task_id=None, *, eou_mode: bool = False, tmp_dir: Path | None = None, allow_fallback: bool = True, audit_path: Path | None = None):
	"""
	Transcribe with a NeMo ASRModel, skipping already-completed chunks,
	and write per-chunk + combined transcripts.
	"""
	transcripts_dir = Path("transcripts")
	transcripts_dir.mkdir(exist_ok=True)
	clean_parts: list[Path] = []
	ts_parts: list[Path] = []
	total = len(chunks)
	suspicious_empty = 0
	empty_ok = 0
	recovered_low_coverage = 0
	# store hypotheses for adjudication (utterance id -> text)
	hyp_baseline: dict[str, str] = {}
	hyp_recover: dict[str, str] = {}
	hyp_alt_asr: dict[str, str] = {}  # future: distil-large-v3.5
	utt_dur: dict[str, float] = {}

	# If a shared progress bar is provided, configure the inner task for this file.
	if progress is not None and chunk_task_id is not None:
		progress.update(chunk_task_id, total=total, completed=0, description=_progress_desc(base, 0, total, "NeMo"))

	# Word timestamps are not supported in a useful way for EOU-style streaming models.
	# We keep the flag for interface consistency but only write clean text.
	if eou_mode and timestamps:
		print("⚠️  Note: word-level timestamps are not supported for the realtime EOU CPU model; writing transcript text only.")
		timestamps = False

	for idx, c in enumerate(chunks):
		chunk_txt, chunk_ts, combined_txt, combined_ts = _chunk_paths(base, idx)

		flags: list[str] = []

		# Needed for duration mapping + quality checks later
		meta = wav_info(Path(c))

		used_recover = False
		used_adjudicate = False
		# Compute once up-front so all branches can use it reliably
		rms = wav_rms(Path(c))

		# Per-chunk hypothesis/resume layout
		hpaths = _hypothesis_paths(base, idx)
		hpaths["dir"].mkdir(parents=True, exist_ok=True)

		# If adjudicated exists, we can treat this chunk as complete (even if chunk_txt exists).
		if cfg.adjudicate and hpaths["adjudicated_txt"].exists() and _is_usable_text_file(hpaths["adjudicated_txt"]):
			# still include in stitching
			clean_parts.append(hpaths["adjudicated_txt"])
			if progress is not None and chunk_task_id is not None:
				progress.advance(chunk_task_id, 1)
				progress.update(chunk_task_id, description=_progress_desc(base, idx, total, "NeMo"))
			continue

		need_clean = not _is_usable_text_file(chunk_txt)
		need_ts = timestamps and (not chunk_ts.exists())

		# ── Resume-safe adjudication: if chunk files exist but adjudication is missing,
		# and we have hypotheses on disk, adjudicate without re-transcribing.
		if cfg.adjudicate and (not hpaths["adjudicated_txt"].exists()):
			has_any_hyp = any(p.exists() and _is_usable_text_file(p) for p in (hpaths["baseline"], hpaths["recover"], hpaths["alt_asr"]))
			if has_any_hyp:
				h = {
					"baseline": hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["baseline"].exists() else "",
					"recover": hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["recover"].exists() else "",
					"alt_asr": hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["alt_asr"].exists() else "",
				}
				multi = sum(1 for v in h.values() if _has_words(v)) >= 2
				should = multi and (not cfg.adjudicate_only_when_flagged)  # we don't know flags here; treat as manual mode
				if should:
					prompt = _build_adjudication_prompt(h, chunk_seconds=float(meta.get("duration_s", 0.0) or 0.0), flags=["resume_adjudicate"])
					j = _run_llama_cpp_json(cfg, prompt)
					final_text = (str(j.get("final", "")).strip() if isinstance(j, dict) else "") or _choose_fallback_best(h)
					if _has_words(final_text):
						hpaths["adjudicated_json"].write_text(json.dumps(j, ensure_ascii=False, indent=2) + "\n", encoding="utf-8") if isinstance(j, dict) else None
						hpaths["adjudicated_txt"].write_text(final_text + "\n", encoding="utf-8")
						clean_parts.append(hpaths["adjudicated_txt"])
						if progress is not None and chunk_task_id is not None:
							progress.advance(chunk_task_id, 1)
						continue

		if progress is not None and chunk_task_id is not None:
			progress.update(chunk_task_id, description=f"NeMo {base}  chunk {idx+1}/{total} (processing)")
 
		if not need_clean and not need_ts:
			# Keep it quiet during progress; still resumable
			clean_parts.append(chunk_txt)
			if timestamps:
				ts_parts.append(chunk_ts)
			if audit_path is not None:
				# Audit even when skipping (resume mode)
				try:
					existing = chunk_txt.read_text(encoding="utf-8", errors="ignore")
				except Exception:
					existing = ""

				vad = vad_stats(Path(c), aggressiveness=3, frame_ms=30)
				vad_likely = vad_likely_speech(Path(c))
				st = transcript_stats(existing)
				rec = {
					"base": base,
					"chunk_index": idx,
					"chunk_wav": str(c),
					"chunk_txt": str(chunk_txt),
					"audio": meta,
					"rms": rms,
					"vad": vad,
					"vad_likely_speech": vad_likely,
					"transcript": st,
					"skipped": True,
					"used_recover": used_recover,
					"used_adjudicate": used_adjudicate,
					"hyp_baseline_path": str(hpaths["baseline"]),
					"hyp_recover_path": str(hpaths["recover"]) if used_recover else "",
					"hyp_alt_asr_path": str(hpaths["alt_asr"]) if _is_usable_text_file(hpaths["alt_asr"]) else "",
					"adjudicated_path": str(hpaths["adjudicated_txt"]) if used_adjudicate else "",
					"flags": flags,	
				}
				audit_path.parent.mkdir(parents=True, exist_ok=True)
				with audit_path.open("a", encoding="utf-8") as f:
					f.write(json.dumps(rec, ensure_ascii=False) + "\n")
			if progress is not None and chunk_task_id is not None:
				progress.advance(chunk_task_id, 1)
			continue

		utt_id = f"{base}_{idx:03d}"
		utt_dur[utt_id] = float(meta.get("duration_s", 0.0) or 0.0)

		# If not using shared rich progress, show normal prints
		if progress is None:
			print(f"🔊 Transcribing chunk {c.name} → {chunk_txt.name}")

		# CPU realtime EOU models often emit only the first utterance for long audio.
		# For those, sub-chunk the audio (e.g., 30s) and concatenate.
		if eou_mode:
			# keep subchunks under tmp_dir so cleanup is automatic
			_sub_root = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks")) / "_eou_subchunks" / f"{base}_{idx:03d}"
			subs = split_wav_fixed(c, _sub_root, segment_s=30)
			texts: list[str] = []
			for s in subs:
				# Skip extremely short segments (can cause NeMo to return [])
				try:
					import soundfile as sf
					info = sf.info(str(s))
					if info.frames == 0 or (info.frames / float(info.samplerate)) < 0.2:
						continue
				except Exception:
					pass

				t = ""
				try:
					try:
						out = model.transcribe([str(s)], timestamps=False, verbose=False)
					except TypeError:
						out = model.transcribe([str(s)], timestamps=False)

					if out:
						r = out[0]
						t = getattr(r, "text", r) or ""
				except Exception:
					t = ""

				t = t.replace("<EOU>", "").strip()
				if t:
					texts.append(t)
			result_text = " ".join(texts).strip()
			result_ts_words = []
		else:
			result_text = ""
			result_ts_words = []
			try:
				try:
					out = model.transcribe([str(c)], timestamps=timestamps, verbose=False)
				except TypeError:
					out = model.transcribe([str(c)], timestamps=timestamps)
			except Exception:
				out = None

			if out:
				result = out[0]
				result_text = (result.text or "")
				result_ts_words = (result.timestamp.get("word", []) if getattr(result, "timestamp", None) else [])
				# Persist baseline hypothesis
				if _has_words(result_text):
					hyp_baseline[utt_id] = result_text.strip()
					hpaths["baseline"].write_text(result_text.strip() + "\n", encoding="utf-8")
 
				# GPU-only quality guard: detect "low coverage" transcripts (speech dropped mid-chunk).
				# Example: 5 minutes with only ~100 words often means minutes were missed.
				if not allow_fallback:
					wc = len((result_text or "").strip().split())
					wpm = words_per_minute(wc, float(meta.get("duration_s", 0.0) or 0.0))

					# Tuneable: below this is suspicious for conversational speech.
					# Much stricter: only rescue chunks that are *clearly* missing lots of speech.
					# This protects poetic/chant/music-heavy sections where long-context decoding is better.
					MIN_WPM_FOR_RESCUE = 35.0

					if wpm > 0.0 and wpm < MIN_WPM_FOR_RESCUE:
						_tmp = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks"))

						# Probe with SAME model to confirm speech is present across the chunk.
						# If probes indicate text exists, do targeted recovery subchunks.
						if nemo_probe_has_text(model, Path(c), _tmp, probe_dur_s=15.0):
							flags.append("low_wpm_probe_confirmed")
							_sub_root = _tmp / "_gpu_recover_subchunks" / f"{base}_{idx:03d}"
							subs = split_wav_fixed(c, _sub_root, segment_s=30)
							texts: list[str] = []
							for s in subs:
								try:
									o2 = model.transcribe([str(s)], timestamps=False, verbose=False)
								except Exception:
									o2 = None
								if o2:
									t2 = (getattr(o2[0], "text", "") or "").strip()
									if t2:
										texts.append(t2)
							recovered = " ".join(texts).strip()
							if recovered:
								hyp_recover[utt_id] = recovered
								hpaths["recover"].write_text(recovered + "\n", encoding="utf-8")
								used_recover = True
								recovered_low_coverage += 1

				# GPU strict mode: do NOT abort on empty text.
				# Empty text can be correct for music/no-speech chunks; VAD can be wrong.
				# Instead, if empty, probe with the SAME model. If probes show text,
				# recover by transcribing 30s subchunks (only for this chunk).
				if (not allow_fallback) and (not _has_words(result_text)):
					_tmp = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks"))
					has_text = nemo_probe_has_text(model, Path(c), _tmp, probe_dur_s=15.0)
					if has_text:
						flags.append("empty_probe_confirmed")
						suspicious_empty += 1
						# Targeted recovery: 30s subchunks for this one problematic chunk
						_sub_root = _tmp / "_gpu_recover_subchunks" / f"{base}_{idx:03d}"
						subs = split_wav_fixed(c, _sub_root, segment_s=30)
						texts: list[str] = []
						for s in subs:
							try:
								try:
									o2 = model.transcribe([str(s)], timestamps=False, verbose=False)
								except TypeError:
									o2 = model.transcribe([str(s)], timestamps=False)
							except Exception:
								o2 = None
							if o2:
								r2 = o2[0]
								t2 = (getattr(r2, "text", "") or "").strip()
								if t2:
									texts.append(t2)
						result_text = " ".join(texts).strip()
						result_ts_words = []
					else:
						empty_ok += 1
			else:
				# IMPORTANT: On GPU we must never fall back. On CPU, fallback is allowed.
				if not allow_fallback:
					# Same handling as empty text: do not abort; audit + continue.
					result_text = ""
					result_ts_words = []
					_tmp = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks"))
					has_text = nemo_probe_has_text(model, Path(c), _tmp, probe_dur_s=15.0)
					if has_text:
						suspicious_empty += 1
					else:
						empty_ok += 1
				else:
					# CPU mode: no Whisper fallback in this architecture.
					# Keep empty and let audit/logging surface the failure.
					flags.append("cpu_empty_no_fallback")
					result_text = ""
					result_ts_words = []

		# ── Adjudication (semantic, extractive-only)
		final_text = (result_text or "").strip()
		if cfg.adjudicate:
			h = {
				"baseline": hyp_baseline.get(utt_id, "") or (hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["baseline"].exists() else ""),
				"recover": hyp_recover.get(utt_id, "") or (hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["recover"].exists() else ""),
				"alt_asr": hyp_alt_asr.get(utt_id, "") or (hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["alt_asr"].exists() else ""),
			}
			multi = sum(1 for v in h.values() if _has_words(v)) >= 2
			should = multi and (not cfg.adjudicate_only_when_flagged or bool(flags))
			if should:
				prompt = _build_adjudication_prompt(h, chunk_seconds=float(meta.get("duration_s", 0.0) or 0.0), flags=flags)
				j = _run_llama_cpp_json(cfg, prompt)
				if isinstance(j, dict) and _has_words(str(j.get("final", ""))):
					final_text = str(j.get("final", "")).strip()
					used_adjudicate = True
					hpaths["adjudicated_json"].write_text(json.dumps(j, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
					hpaths["adjudicated_txt"].write_text(final_text + "\n", encoding="utf-8")
				else:
					final_text = _choose_fallback_best(h) or final_text

		# Prefer adjudicated output for stitching if available
		if used_adjudicate and _is_usable_text_file(hpaths["adjudicated_txt"]):
			clean_parts.append(hpaths["adjudicated_txt"])
		else:
			# Write final text to chunk txt (keeps existing stitcher behavior)
			if need_clean:
				chunk_txt.write_text(final_text + "\n", encoding="utf-8")
			clean_parts.append(chunk_txt)

		# Ensure chunk_txt contains the final chosen text (not overwritten by baseline).
		# If adjudication ran, final_text may differ from result_text.
		if (not used_adjudicate) and need_clean:
			# already written above; this block is intentionally a no-op
			pass

		if audit_path is not None:
			meta = wav_info(Path(c))
			vad = vad_stats(Path(c), aggressiveness=3, frame_ms=30)
			vad_likely = vad_likely_speech(Path(c))
			st = transcript_stats(final_text or "")
			wpm = words_per_minute(st["words"], float(meta.get("duration_s", 0.0) or 0.0))
			rec = {
				"base": base,
				"chunk_index": idx,
				"chunk_wav": str(c),
				"chunk_txt": str(chunk_txt),
				"audio": meta,
				"rms": rms,
				"used_recover": used_recover,
				"used_adjudicate": used_adjudicate,
				"wpm": wpm,
				"vad": vad,
				"vad_likely_speech": vad_likely,
				"transcript": st,
				"hyp_baseline_path": str(hpaths["baseline"]),
				"hyp_recover_path": str(hpaths["recover"]) if used_recover else "",
				"hyp_alt_asr_path": str(hpaths["alt_asr"]) if _is_usable_text_file(hpaths["alt_asr"]) else "",
				"adjudicated_path": str(hpaths["adjudicated_txt"]) if used_adjudicate else "",
				"flags": flags,
			}
			audit_path.parent.mkdir(parents=True, exist_ok=True)
			with audit_path.open("a", encoding="utf-8") as f:
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")

		# If timestamps requested, write them to a separate file
		if need_ts:
			lines = []
			for w in result_ts_words:
				lines.append(f"{w['start']:.2f}s→{w['end']:.2f}: {w['word']}\n")
			chunk_ts.write_text("".join(lines), encoding="utf-8")
 
		if progress is None:
			print(f"✅ Saved: {chunk_txt.name}" + (f" + {chunk_ts.name}" if need_ts else ""))
 
		# NOTE: clean_parts/ts_parts are already appended above (avoid duplicates).
		if timestamps and need_ts:
			ts_parts.append(chunk_ts)

		if progress is not None and chunk_task_id is not None:
			# ✅ advance after each processed chunk so the bar actually fills
			progress.advance(chunk_task_id, 1)
			progress.update(chunk_task_id, description=_progress_desc(base, idx, total, "NeMo"))

	# File-level summary (GPU only)
	if (not allow_fallback) and (suspicious_empty or empty_ok):
		print(f"⚠️  {base}: empty chunks (ok={empty_ok}, suspicious={suspicious_empty}), rescued_low_coverage={recovered_low_coverage}. See audit for details.")

	# Combine from per-chunk outputs (resumable and stable).
	# NOTE: clean_parts should already prefer adjudicated outputs when available.
	_combine_files(combined_txt, clean_parts)
	
	print(f"📄 Combined transcript: {combined_txt}")
	if timestamps:
		_combine_files(combined_ts, ts_parts)
		print(f"🕒 Combined timestamps: {combined_ts}")

def _is_tty() -> bool:
	# Conservative: require both stdin and stdout to be a TTY for interactive UI.
	return sys.stdin.isatty() and sys.stdout.isatty()


def interactive_flow(cfg: AppConfig):
	"""
	Interactive "TUI-like" flow for selecting files and options.

	Returns:
		(audio_paths: list[str], timestamps: bool)
	"""
	print("Loading interactive mode…")

	# Lazy imports so normal CLI usage doesn't require UI deps.
	try:
		from InquirerPy import inquirer
		from InquirerPy.separator import Separator
	except Exception:
		print(
			"❌ Interactive mode requires InquirerPy.\n"
			"   Install it in your venv:\n"
			"	 pip install InquirerPy rich\n",
			file=sys.stderr,
		)
		raise

	try:
		from rich.console import Console
		from rich.panel import Panel
	except Exception:
		print(
			"❌ Interactive mode requires rich.\n"
			"   Install it in your venv:\n"
			"	 pip install rich\n",
			file=sys.stderr,
		)
		raise

	console = Console()

	console.print(Panel.fit(
		"[bold]Transcriber[/bold]\n"
		"Select audio files and options, then start transcription.",
		title="Interactive Mode",
	))

	# Gather candidate files from audio/ (common extensions)
	exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv"}
	candidates = []
	project_audio_dir = Path(cfg.audio_dir)
	if project_audio_dir.exists():
		for p in sorted(project_audio_dir.rglob("*")):
			if p.is_file() and p.suffix.lower() in exts:
				# display relative path for readability
				candidates.append(p)

	choices = []
	if candidates:
		choices.append(Separator("— audio/ files —"))
		for p in candidates:
			choices.append(str(p))
	else:
		choices.append(Separator("— No audio files found under audio/ —"))

	choices.append(Separator("— Other —"))
	choices.append("[Pick one or more from audio/]")
	choices.append("[Enter a path manually]")
	choices.append("[Enter multiple paths manually]")
	choices.append("[Settings / Defaults]")

	selection = _safe_execute(inquirer.select(
		message="Choose how you want to provide input:",
		choices=choices,
		default="[Pick one or more from audio/]" if candidates else "[Enter a path manually]",

	))

	if selection == "[Settings / Defaults]":
		# simple settings editor
		new_audio_dir = _safe_execute(inquirer.text(
			message="Default audio directory:",
			default=cfg.audio_dir
		)).strip() or cfg.audio_dir

		new_timestamps = _safe_execute(inquirer.confirm(
			message="Default to timestamps ON?",
			default=cfg.default_timestamps
		))

		new_progress = _safe_execute(inquirer.confirm(
			message="Show progress bars by default?",
			default=cfg.show_progress
		))

		new_enhance = _safe_execute(inquirer.confirm(
			message="Enable speech enhancement (ffmpeg filters) by default?",
			default=getattr(cfg, "enhance_speech", False)
		))

		new_demucs = _safe_execute(inquirer.confirm(
			message="Enable Demucs vocals-only preprocessing by default? (slower)",
			default=getattr(cfg, "use_demucs", False)
		))

		new_demucs_model = _safe_execute(inquirer.text(
			message="Demucs model name:",
			default=getattr(cfg, "demucs_model", "htdemucs")
		)).strip() or getattr(cfg, "demucs_model", "htdemucs")

		cfg.audio_dir = new_audio_dir
		cfg.default_timestamps = new_timestamps
		cfg.show_progress = new_progress
		cfg.enhance_speech = new_enhance
		cfg.use_demucs = new_demucs
		cfg.demucs_model = new_demucs_model

		# Adjudication overrides
		if args.no_adjudicate:
			cfg.adjudicate = False
		if args.llama_bin:
			cfg.llama_bin = args.llama_bin
		if args.llama_model:
			cfg.llama_model = args.llama_model
		if args.llama_grammar is not None:
			cfg.llama_grammar = args.llama_grammar

		save_config(cfg)
		console.print("[green]Saved settings.[/green]\n")
		# Restart the flow with updated settings
		return interactive_flow(cfg)

	audio_paths: list[str] = []

	if selection == "[Pick one or more from audio/]":
		picked = _safe_execute(inquirer.checkbox(
			message="Select one or more files to transcribe (Space to toggle):",
			choices=[str(p) for p in candidates],
		))
		audio_paths = [str(x) for x in picked]

	if selection == "[Enter a path manually]":
		p = _safe_execute(inquirer.text(message="Enter a path to an audio file:")).strip()
		if p:
			audio_paths = [p]
	elif selection == "[Enter multiple paths manually]":
		console.print("Paste paths separated by spaces OR one per line. End with an empty line.")
		lines = []
		try:
			while True:
				line = input("> ").strip()
				if not line:
					break
				lines.append(line)
		except KeyboardInterrupt:
			console.print("\n[red]Closing.[/red]")
			raise SystemExit(130)
		# Support either whitespace-separated on one line or one-per-line.
		raw = " ".join(lines).strip()
		audio_paths = raw.split() if raw else []
	elif selection and selection not in (
		"[Enter a path manually]",
		"[Enter multiple paths manually]",
		"[Pick one or more from audio/]",
		"[Settings / Defaults]",
	):
		# They selected a specific file from the list → just use it (no second prompt).
		audio_paths = [selection]

	if not audio_paths:
		console.print("[red]No audio files selected.[/red]")
		sys.exit(2)

	# Validate files early to avoid ffprobe tracebacks later
	bad = []
	good = []
	for p in audio_paths:
		pp = Path(p)
		if not pp.exists() or not pp.is_file():
			bad.append(p)
		else:
			good.append(str(pp))
	if bad:
		console.print("[red]These paths do not exist or are not files:[/red]")
		for b in bad:
			console.print(f"  • {b}")
		sys.exit(2)
	audio_paths = good

	timestamps = _safe_execute(inquirer.confirm(message="Include word-level timestamps?", default=cfg.default_timestamps))

	console.print("\n[bold]Ready:[/bold]")
	for p in audio_paths:
		console.print(f"  • {p}")
	console.print(f"  • timestamps: {timestamps}\n")

	if not _safe_execute(inquirer.confirm(message="Start transcription now?", default=True)):
		console.print("Closing.")
		sys.exit(0)

	return audio_paths, timestamps

def run_batch_with_progress(prepared: dict, mode_name: str, per_file_fn, show_progress: bool = True, progress=None):
	"""
	Run per_file_fn(base, chunks, progress, chunk_task_id) over all prepared items,
	with an outer (files) and inner (chunks) Rich progress display when available.
	"""
	total_files = len(prepared)
	# Reuse a shared progress instance if provided (prevents multiple progress UIs)
	if progress is None:
		progress_ctx, progress = _maybe_rich_progress(enabled=show_progress)
	else:
		progress_ctx = nullcontext()

	if progress is None:
		# No rich available → just run normally
		for base, chunks in prepared.items():
			per_file_fn(base, chunks, None, None)
		return

	with progress_ctx:
		files_task = progress.add_task(f"{mode_name} files", total=total_files)
		chunks_task = progress.add_task(f"{mode_name} chunks", total=1)

		for base, chunks in prepared.items():
			# Reset inner task for this file
			progress.update(chunks_task, total=len(chunks), completed=0, description=f"{mode_name} {base} chunks")

			per_file_fn(base, chunks, progress, chunks_task)

			# Mark file done
			progress.advance(files_task, 1)

		# Clean up tasks so mode switches (NeMo→Whisper) don't leave old bars behind
		progress.remove_task(chunks_task)
		progress.remove_task(files_task)

def main():
	parser = argparse.ArgumentParser(
		description="Transcribe audio with GPU (Parakeet) or CPU (Whisper) fallback"
	)
	parser.add_argument("audio", nargs="*", help="Path(s) to audio files")
	parser.add_argument(
		"--timestamps", action="store_true", help="Include word-level timestamps"
	)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="Run interactive terminal UI to select files and options",
	)
	parser.add_argument(
		"--cli",
		action="store_true",
		help="Force non-interactive CLI mode (argparse only)",
	)
	parser.add_argument("--audit", action="store_true", help="Write per-chunk audit info to transcripts/audit/<base>_audit.jsonl")
	parser.add_argument("--enhance-speech", action="store_true", help="Apply ffmpeg speech enhancement filters before splitting (optional)")
	parser.add_argument("--demucs", action="store_true", help="Use Demucs to extract vocals before splitting (optional, slower)")
	parser.add_argument("--demucs-model", default=None, help="Demucs model name (default from config: htdemucs)")
	parser.add_argument("--no-adjudicate", action="store_true", help="Disable local LLM adjudication stage")
	parser.add_argument("--llama-bin", default=None, help="llama.cpp binary name/path (default from config)")
	parser.add_argument("--llama-model", default=None, help="GGUF model path for adjudicator (default from config)")
	parser.add_argument("--llama-grammar", default=None, help="Optional GBNF grammar file path for JSON-only output")


	args = parser.parse_args()
	cfg = load_config()

	import logging
	logging.getLogger("nemo").setLevel(logging.ERROR)

	# Decide mode:
	# - If user passed audio args: default to CLI behavior.
	# - If no audio args: default to interactive (TTY only).
	# - --interactive forces interactive.
	# - --cli forces non-interactive.
	if args.cli and args.interactive:
		print("❌ Choose only one of --cli or --interactive.", file=sys.stderr)
		sys.exit(2)

	if args.cli:
		if not args.audio:
			print("❌ CLI mode requires one or more audio paths.", file=sys.stderr)
			parser.print_usage()
			sys.exit(2)
	elif args.interactive or (not args.audio):
		if not _is_tty():
			print(
				"❌ Interactive mode requires a TTY (real terminal).\n"
				"   Provide audio paths as arguments, or run in a normal terminal.",
				file=sys.stderr,
			)
			sys.exit(2)
		# Run interactive flow; overwrite args with user selections.
		audio_paths, timestamps = interactive_flow(cfg)
		args.audio = audio_paths
		args.timestamps = timestamps

	tmp_dir = Path(".tmp_audio_chunks")
	tmp_dir.mkdir(exist_ok=True)

	# Resolve preprocessing toggles:
	# CLI flags override config; otherwise use config defaults.
	enhance_speech = bool(args.enhance_speech) or bool(getattr(cfg, "enhance_speech", False))
	use_demucs = bool(args.demucs) or bool(getattr(cfg, "use_demucs", False))
	demucs_model = args.demucs_model or getattr(cfg, "demucs_model", "htdemucs")

	# 1) Prepare inputs
	paths = [Path(p) for p in args.audio]

	progress_ctx, shared_progress = _maybe_rich_progress(enabled=cfg.show_progress)
	with progress_ctx:
		prepared = prepare_inputs_with_progress(
			paths,
			tmp_dir,
			show_progress=cfg.show_progress,
			progress=shared_progress,
			enhance_speech=enhance_speech,
			use_demucs=use_demucs,
			demucs_model=demucs_model,
		)
		gpu_success = False

		# ── GPU path → parakeet-tdt-0.6b-v3
		import torch

		if torch.cuda.is_available():
			device = torch.device("cuda")
			print(f"✅ GPU detected → using parakeet-tdt-0.6b-v3 on {device}")
			try:
				import nemo.collections.asr as nemo_asr  # lazy import
				model = nemo_asr.models.ASRModel.from_pretrained(
					model_name="nvidia/parakeet-tdt-0.6b-v3",
					map_location=device,
				).to(device)
				def _nemo_gpu_file(base, chunks, progress, chunk_task_id):
					audit_path = None
					if args.audit:
						audit_path = Path("transcripts") / "audit" / f"{base}_audit.jsonl"
		
					# GPU strict mode: never fall back, never EOU, no subchunking.
					return transcribe_chunks(
						base,
						chunks,
						model,
						args.timestamps,
						progress=progress,
						chunk_task_id=chunk_task_id,
						eou_mode=False,
						tmp_dir=tmp_dir,
						allow_fallback=False,
						audit_path=audit_path,
					)

				run_batch_with_progress(prepared, "NeMo(GPU)", _nemo_gpu_file, show_progress=cfg.show_progress, progress=shared_progress)
				gpu_success = True
			except Exception as e:
				# GPU present but GPU path failed: do NOT fall back. Exit loudly.
				print(f"❌ GPU NeMo transcription failed: {e}")
				print("   This system has CUDA, so we refuse CPU/Whisper fallback by design.")
				print("   Suggested diagnostics:")
				print("   - Confirm the model loads: try running a short file and watch for NeMo errors")
				print("   - Check VRAM usage (nvidia-smi) and CUDA compatibility")
				print("   - If this happens on a specific chunk, delete that chunk txt and rerun to reproduce")
				raise SystemExit(2)

		# ── CPU-only path (no CUDA) → parakeet_realtime_eou_120m-v1, then Whisper
		if not torch.cuda.is_available():
			print("⚠️ No GPU transcription → using parakeet_realtime_eou_120m-v1 on CPU")
			import nemo.collections.asr as nemo_asr
			cpu_model_name = "nvidia/parakeet_realtime_eou_120m-v1"
			is_eou = "realtime_eou" in cpu_model_name

			nemo_retries = getattr(cfg, "nemo_retries", 3)
			last_err = None
			for attempt in range(1, nemo_retries + 1):
				try:
					cpu_model = nemo_asr.models.ASRModel.from_pretrained(
						model_name=cpu_model_name,
						map_location="cpu",
					)

					def _nemo_cpu_file(base, chunks, progress, chunk_task_id):
						audit_path = None
						if args.audit:
							audit_path = Path("transcripts") / "audit" / f"{base}_audit.jsonl"

						return transcribe_chunks(
							base, chunks, cpu_model, args.timestamps,
							progress=progress, chunk_task_id=chunk_task_id,
							eou_mode=is_eou, tmp_dir=tmp_dir,
							allow_fallback=True,
							audit_path=audit_path,
						)

					run_batch_with_progress(
						prepared, f"NeMo(CPU) try {attempt}/{nemo_retries}",
						_nemo_cpu_file, show_progress=cfg.show_progress, progress=shared_progress
					)
					last_err = None
					break
				except Exception as e:
					last_err = e
					print(f"⚠ NeMo(CPU) attempt {attempt}/{nemo_retries} failed: {e}")
					# Resume logic means re-running will skip completed chunk files.

			if last_err is not None:
				print(f"❌ NeMo(CPU) failed after {nemo_retries} attempts. No Whisper fallback in this architecture.")
				raise SystemExit(2)

	# 4) Clean up
	# ── Combine all per-file transcripts into one master transcript ──────────
	master_path = Path("transcripts") / "ALL_TRANSCRIPTS.txt"
	print(f"\n📚 Writing master transcript: {master_path.name}")
	with open(master_path, "w", encoding="utf-8") as mf:
		# iterate in the same order as user passed audio files
		for audio_file in args.audio:
			base = Path(audio_file).stem
			combined = Path("transcripts") / f"{base}.txt"
			if not combined.exists():
				print(f"⚠️  Missing transcript for {base}, skipping")
				continue
			mf.write(f"==== {base} ====\n")
			mf.write(combined.read_text(encoding="utf-8"))
			mf.write("\n\n")
	print("✅ Master transcript complete.")

	# ── If timestamps were requested, also build a master timestamps file ─────
	if args.timestamps:
		master_ts_path = Path("transcripts") / "ALL_TIMESTAMPS.txt"
		print(f"\n🕒 Writing master timestamps: {master_ts_path.name}")
		with open(master_ts_path, "w", encoding="utf-8") as tf:
			for audio_file in args.audio:
				base = Path(audio_file).stem
				combined_ts = Path("transcripts") / f"{base}.timestamps.txt"
				if not combined_ts.exists():
					print(f"⚠️  Missing timestamps for {base}, skipping")
					continue
				tf.write(f"==== {base} ====\n")
				tf.write(combined_ts.read_text(encoding="utf-8"))
				tf.write("\n\n")
		print("✅ Master timestamps complete.")

	# now remove temp chunks
	shutil.rmtree(tmp_dir)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n⛔ Closing (Ctrl+C).")
		raise SystemExit(130)
