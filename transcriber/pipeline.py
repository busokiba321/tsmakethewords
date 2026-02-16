from __future__ import annotations

import json
from pathlib import Path

from .adjudicate import adjudicate_chunk, build_prompt, run_llama_cpp_json, choose_fallback_best
from .audio import nemo_probe_has_text, split_wav_fixed, vad_likely_speech, vad_stats, wav_info, wav_rms
from .config import AppConfig
from .hypotheses import hypothesis_paths
from .ui import progress_desc
from .utils import combine_files, has_words, is_usable_text_file, transcript_stats, words_per_minute


def chunk_paths(base: str, idx: int):
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(exist_ok=True)
    clean_dir = transcripts_dir / "clean"
    ts_dir = transcripts_dir / "timestamps"
    clean_dir.mkdir(exist_ok=True)
    ts_dir.mkdir(exist_ok=True)
    chunk_txt = clean_dir / f"{base}_{idx:03d}.txt"
    chunk_ts = ts_dir / f"{base}_{idx:03d}.txt"
    combined_txt = transcripts_dir / f"{base}.txt"
    combined_ts = transcripts_dir / f"{base}_timestamps.txt"
    return chunk_txt, chunk_ts, combined_txt, combined_ts


def transcribe_chunks(
    base,
    chunks,
    model,
    timestamps,
    cfg: AppConfig,
    progress=None,
    chunk_task_id=None,
    *,
    eou_mode: bool = False,
    tmp_dir: Path | None = None,
    allow_fallback: bool = True,
    audit_path: Path | None = None,
):
    clean_parts: list[Path] = []
    ts_parts: list[Path] = []
    total = len(chunks)
    suspicious_empty = 0
    empty_ok = 0
    recovered_low_coverage = 0

    if progress is not None and chunk_task_id is not None:
        progress.update(chunk_task_id, total=total, completed=0, description=progress_desc(base, 0, total, "NeMo"))

    for idx, c in enumerate(chunks):
        c = Path(c)
        chunk_txt, chunk_ts, combined_txt, combined_ts = chunk_paths(base, idx)
        meta = wav_info(c)
        flags: list[str] = []
        used_recover = False
        used_adjudicate = False
        rms = wav_rms(c)

        hpaths = hypothesis_paths(base, idx)
        hpaths["dir"].mkdir(parents=True, exist_ok=True)

        if cfg.adjudicate and is_usable_text_file(hpaths["adjudicated_txt"]):
            clean_parts.append(hpaths["adjudicated_txt"])
            if progress is not None and chunk_task_id is not None:
                progress.advance(chunk_task_id, 1)
            continue

        need_clean = not is_usable_text_file(chunk_txt)
        need_ts = timestamps and (not chunk_ts.exists())

        # Resume-safe adjudication from existing hypotheses without re-running ASR.
        if cfg.adjudicate and not hpaths["adjudicated_txt"].exists():
            has_any_hyp = any(is_usable_text_file(p) for p in (hpaths["baseline"], hpaths["recover"], hpaths["alt_asr"]))
            if has_any_hyp:
                h = {
                    "baseline": hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["baseline"].exists() else "",
                    "recover": hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["recover"].exists() else "",
                    "alt_asr": hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["alt_asr"].exists() else "",
                }
                multi = sum(1 for v in h.values() if has_words(v)) >= 2
                if multi and (not cfg.adjudicate_only_when_flagged):
                    prompt = build_prompt(h, chunk_seconds=float(meta.get("duration_s", 0.0) or 0.0), flags=["resume_adjudicate"])
                    j = run_llama_cpp_json(cfg, prompt)
                    final = (str(j.get("final", "")).strip() if isinstance(j, dict) else "") or choose_fallback_best(h)
                    if has_words(final):
                        if isinstance(j, dict):
                            hpaths["adjudicated_json"].write_text(json.dumps(j, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        hpaths["adjudicated_txt"].write_text(final + "\n", encoding="utf-8")
                        clean_parts.append(hpaths["adjudicated_txt"])
                        if progress is not None and chunk_task_id is not None:
                            progress.advance(chunk_task_id, 1)
                        continue

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
            result_text = (getattr(out[0], "text", "") or "")
            result_ts_words = (out[0].timestamp.get("word", []) if getattr(out[0], "timestamp", None) else [])
            if has_words(result_text):
                hpaths["baseline"].write_text(result_text.strip() + "\n", encoding="utf-8")

        if not allow_fallback:
            wc = len((result_text or "").strip().split())
            wpm = words_per_minute(wc, float(meta.get("duration_s", 0.0) or 0.0))
            if wpm > 0.0 and wpm < 35.0:
                _tmp = tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks")
                if nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0):
                    flags.append("low_wpm_probe_confirmed")
                    subs = split_wav_fixed(c, _tmp / "_gpu_recover_subchunks" / f"{base}_{idx:03d}", segment_s=30)
                    texts = []
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
                        hpaths["recover"].write_text(recovered + "\n", encoding="utf-8")
                        used_recover = True
                        recovered_low_coverage += 1

        if (not allow_fallback) and (not has_words(result_text)):
            _tmp = tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks")
            if nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0):
                flags.append("empty_probe_confirmed")
                suspicious_empty += 1
            else:
                empty_ok += 1

        final_text = (result_text or "").strip()
        if cfg.adjudicate:
            h = {
                "baseline": hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["baseline"].exists() else "",
                "recover": hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["recover"].exists() else "",
                "alt_asr": hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["alt_asr"].exists() else "",
            }
            final_text, j, used_adjudicate = adjudicate_chunk(cfg, h, chunk_seconds=float(meta.get("duration_s", 0.0) or 0.0), flags=flags)
            if used_adjudicate and isinstance(j, dict):
                hpaths["adjudicated_json"].write_text(json.dumps(j, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                hpaths["adjudicated_txt"].write_text(final_text + "\n", encoding="utf-8")

        if used_adjudicate and is_usable_text_file(hpaths["adjudicated_txt"]):
            clean_parts.append(hpaths["adjudicated_txt"])
        else:
            if need_clean:
                chunk_txt.write_text(final_text + "\n", encoding="utf-8")
            clean_parts.append(chunk_txt)

        if timestamps and need_ts:
            lines = [f"{w['start']:.2f}s→{w['end']:.2f}: {w['word']}\n" for w in result_ts_words]
            chunk_ts.write_text("".join(lines), encoding="utf-8")
            ts_parts.append(chunk_ts)

        if audit_path is not None:
            vad = vad_stats(c, aggressiveness=3, frame_ms=30)
            vad_likely = vad_likely_speech(c)
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
                "hyp_alt_asr_path": str(hpaths["alt_asr"]) if is_usable_text_file(hpaths["alt_asr"]) else "",
                "adjudicated_path": str(hpaths["adjudicated_txt"]) if used_adjudicate else "",
                "flags": flags,
            }
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if progress is not None and chunk_task_id is not None:
            progress.advance(chunk_task_id, 1)

    combine_files(combined_txt, clean_parts)
    if timestamps:
        combine_files(combined_ts, ts_parts)

    if (not allow_fallback) and (suspicious_empty or empty_ok):
        print(f"⚠️  {base}: empty chunks (ok={empty_ok}, suspicious={suspicious_empty}), rescued_low_coverage={recovered_low_coverage}.")

    print(f"📄 Combined transcript: {combined_txt}")
    return combined_txt, combined_ts
