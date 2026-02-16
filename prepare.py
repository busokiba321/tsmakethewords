from __future__ import annotations

from pathlib import Path
from .audio import convert_and_split


def prepare_inputs(paths, tmp_dir: Path, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs", max_duration_s: int = 300):
	prepared = {}
	for p in paths:
		base, chunks = convert_and_split(p, tmp_dir, enhance_speech=enhance_speech, use_demucs=use_demucs, demucs_model=demucs_model, max_duration_s=max_duration_s)
		prepared[base] = chunks
	return prepared


def prepare_inputs_with_progress(paths, tmp_dir: Path, show_progress: bool = True, progress=None, *, enhance_speech: bool = False, use_demucs: bool = False, demucs_model: str = "htdemucs", max_duration_s: int = 300):
	return prepare_inputs(paths, tmp_dir, enhance_speech=enhance_speech, use_demucs=use_demucs, demucs_model=demucs_model, max_duration_s=max_duration_s)
