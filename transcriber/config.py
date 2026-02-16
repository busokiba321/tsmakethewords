from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

	# ── Adjudication (local LLM via llama.cpp)
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
	llama_grammar: str = "grammars/json.gbnf"  # optional path to GBNF grammar file for llama.cpp

	# ── Alternate ASR hypothesis (Faster-Whisper/CT2)
	alt_asr_enabled: bool = True
	alt_asr_model: str = "distil-whisper/distil-large-v3.5-ct2"
	alt_asr_device: str = "cuda"
	alt_asr_compute_type: str = "float16"
	alt_asr_beam_size: int = 5
	alt_asr_vad_filter: bool = False


def load_config() -> AppConfig:
	"""Load config from transcriber.toml (tiny TOML, no third-party deps)."""
	if not CONFIG_PATH.exists():
		return AppConfig()

	import tomllib

	data = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))

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
		llama_grammar=str(data.get("llama_grammar", "grammars/json.gbnf")),
		alt_asr_enabled=bool(data.get("alt_asr_enabled", True)),
		alt_asr_model=str(data.get("alt_asr_model", "distil-whisper/distil-large-v3.5-ct2")),
		alt_asr_device=str(data.get("alt_asr_device", "cuda")),
		alt_asr_compute_type=str(data.get("alt_asr_compute_type", "float16")),
		alt_asr_beam_size=int(data.get("alt_asr_beam_size", 5)),
		alt_asr_vad_filter=bool(data.get("alt_asr_vad_filter", False)),
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
		f"alt_asr_enabled = {str(cfg.alt_asr_enabled).lower()}\n"
		f'alt_asr_model = "{cfg.alt_asr_model}"\n'
		f'alt_asr_device = "{cfg.alt_asr_device}"\n'
		f'alt_asr_compute_type = "{cfg.alt_asr_compute_type}"\n'
		f"alt_asr_beam_size = {int(cfg.alt_asr_beam_size)}\n"
		f"alt_asr_vad_filter = {str(cfg.alt_asr_vad_filter).lower()}\n"
	)
	CONFIG_PATH.write_text(text, encoding="utf-8")
