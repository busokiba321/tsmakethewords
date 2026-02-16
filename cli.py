from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig, load_config, save_config
from .pipeline import transcribe_chunks
from .prepare import prepare_inputs_with_progress
from .ui import maybe_rich_progress, safe_execute


def _is_tty() -> bool:
	return sys.stdin.isatty() and sys.stdout.isatty()


def interactive_flow(cfg: AppConfig):
	print("Loading interactive mode…")
	try:
		from InquirerPy import inquirer
		from InquirerPy.separator import Separator
		from rich.console import Console
		from rich.panel import Panel
	except Exception:
		print("❌ Interactive mode requires InquirerPy + rich.", file=sys.stderr)
		raise

	console = Console()
	console.print(Panel.fit("[bold]Transcriber[/bold]\nSelect audio files and options, then start transcription.", title="Interactive Mode"))

	exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv"}
	candidates = []
	project_audio_dir = Path(cfg.audio_dir)
	if project_audio_dir.exists():
		for p in sorted(project_audio_dir.rglob("*")):
			if p.is_file() and p.suffix.lower() in exts:
				candidates.append(p)

	choices = []
	if candidates:
		choices.append(Separator("── audio/ ──"))
		for p in candidates:
			choices.append({"name": str(p), "value": str(p)})
	else:
		choices.append({"name": "(no audio files found in audio/)", "value": None})

	sel = safe_execute(inquirer.checkbox(message="Select audio files:", choices=choices, validate=lambda r: len([x for x in r if x]) > 0))
	audio_paths = [p for p in sel if p]
	timestamps = safe_execute(inquirer.confirm(message="Write word timestamps?", default=cfg.default_timestamps))

	return audio_paths, timestamps


def run_batch_with_progress(prepared: dict, mode_name: str, per_file_fn, show_progress: bool = True, progress=None):
	if progress is None:
		ctx, prog = maybe_rich_progress(show_progress)
	else:
		ctx, prog = (progress, progress)

	with ctx:
		for base, chunks in prepared.items():
			task_id = prog.add_task(f"{mode_name} {base}", total=len(chunks)) if prog is not None else None
			per_file_fn(base, chunks, prog, task_id)


def main():
	parser = argparse.ArgumentParser(description="Local offline transcription pipeline")
	parser.add_argument("audio", nargs="*", help="Audio file(s) to transcribe")
	parser.add_argument("--timestamps", action="store_true", help="Write word timestamps")
	parser.add_argument("--audit", action="store_true", help="Write per-chunk audit JSONL")
	parser.add_argument("--enhance-speech", action="store_true", help="Apply ffmpeg speech enhancement")
	parser.add_argument("--demucs", action="store_true", help="Use Demucs vocals extraction")
	parser.add_argument("--demucs-model", default=None, help="Demucs model name")

	parser.add_argument("--no-adjudicate", action="store_true", help="Disable adjudication")
	parser.add_argument("--llama-bin", default=None, help="llama.cpp binary")
	parser.add_argument("--llama-model", default=None, help="GGUF adjudicator model")
	parser.add_argument("--llama-grammar", default=None, help="GBNF grammar file")

	args = parser.parse_args()
	cfg = load_config()

	if args.no_adjudicate:
		cfg.adjudicate = False
	if args.llama_bin:
		cfg.llama_bin = args.llama_bin
	if args.llama_model:
		cfg.llama_model = args.llama_model
	if args.llama_grammar is not None:
		cfg.llama_grammar = args.llama_grammar

	enhance_speech = bool(args.enhance_speech) or bool(cfg.enhance_speech)
	use_demucs = bool(args.demucs) or bool(cfg.use_demucs)
	demucs_model = args.demucs_model or cfg.demucs_model

	if not args.audio and _is_tty():
		audio_paths, ts = interactive_flow(cfg)
		args.audio = audio_paths
		args.timestamps = bool(ts)

	if not args.audio:
		print("No audio files specified.", file=sys.stderr)
		raise SystemExit(2)

	paths = [Path(p) for p in args.audio]
	tmp_dir = Path(".tmp_audio_chunks")
	tmp_dir.mkdir(exist_ok=True)

	prepared = prepare_inputs_with_progress(paths, tmp_dir, show_progress=cfg.show_progress, enhance_speech=enhance_speech, use_demucs=use_demucs, demucs_model=demucs_model, max_duration_s=300)

	try:
		import nemo.collections.asr as nemo_asr
		model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
	except Exception as e:
		print(f"❌ Failed to load NeMo model: {e}", file=sys.stderr)
		raise SystemExit(2)

	def _nemo_file(base, chunks, progress, task_id):
		audit_path = (Path("transcripts") / "audit" / f"{base}_audit.jsonl") if args.audit else None
		return transcribe_chunks(base, chunks, model, args.timestamps, cfg, progress=progress, chunk_task_id=task_id, tmp_dir=tmp_dir, allow_fallback=False, audit_path=audit_path)

	run_batch_with_progress(prepared, "NeMo(GPU)", _nemo_file, show_progress=cfg.show_progress)


if __name__ == "__main__":
	main()