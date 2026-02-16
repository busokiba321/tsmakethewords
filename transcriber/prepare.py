from __future__ import annotations

from pathlib import Path

from .audio import MAX_DURATION, convert_and_split, probe
from .ui import maybe_rich_progress


def prepare_inputs(
    paths,
    tmp_dir: Path,
    *,
    enhance_speech: bool = False,
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    max_duration_s: int = MAX_DURATION,
):
    prepared = {}
    for p in paths:
        p = Path(p)
        try:
            cfg = probe(p)
        except Exception as e:
            print(f"❌ Cannot process '{p}': {e}")
            raise SystemExit(2)

        needs = (
            cfg["format"] not in [".wav", ".flac"]
            or cfg["sample_rate"] != 16000
            or cfg["channels"] != 1
            or cfg["duration"] > max_duration_s
        )

        if needs:
            print(f"⚙️  Converting/splitting {p.name}")
            chunks = convert_and_split(
                p,
                tmp_dir,
                enhance_speech=enhance_speech,
                use_demucs=use_demucs,
                demucs_model=demucs_model,
                max_duration_s=max_duration_s,
            )
        else:
            chunks = [p]

        prepared.setdefault(p.stem, []).extend(chunks)
    return prepared


def prepare_inputs_with_progress(
    paths,
    tmp_dir: Path,
    show_progress: bool = True,
    progress=None,
    *,
    enhance_speech: bool = False,
    use_demucs: bool = False,
    demucs_model: str = "htdemucs",
    max_duration_s: int = MAX_DURATION,
):
    if progress is None:
        progress_ctx, pbar = maybe_rich_progress(show_progress)
    else:
        progress_ctx, pbar = progress, progress

    paths = [Path(p) for p in paths]
    if pbar is None:
        return prepare_inputs(
            paths,
            tmp_dir,
            enhance_speech=enhance_speech,
            use_demucs=use_demucs,
            demucs_model=demucs_model,
            max_duration_s=max_duration_s,
        )

    prepared = {}
    with progress_ctx:
        task = pbar.add_task("Preparing audio files", total=len(paths))
        for idx, p in enumerate(paths):
            pbar.update(task, description=f"Preparing {p.name} ({idx + 1}/{len(paths)})")
            prepared.update(
                prepare_inputs(
                    [p],
                    tmp_dir,
                    enhance_speech=enhance_speech,
                    use_demucs=use_demucs,
                    demucs_model=demucs_model,
                    max_duration_s=max_duration_s,
                )
            )
            pbar.advance(task, 1)
        pbar.remove_task(task)
    return prepared
