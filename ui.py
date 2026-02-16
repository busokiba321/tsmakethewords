from __future__ import annotations

import sys
from contextlib import nullcontext


def safe_execute(prompt):
	try:
		return prompt.execute()
	except KeyboardInterrupt:
		print("\n⛔ Closing (CtrlC).")
		raise SystemExit(130)


def maybe_rich_progress(enabled: bool = True):
	if not enabled or not sys.stdout.isatty():
		return nullcontext(), None
	try:
		from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
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


def progress_desc(base: str, idx: int, total: int, mode: str) -> str:
	return f"{mode} {base}  chunk {idx1}/{total}"