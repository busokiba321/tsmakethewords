from __future__ import annotations

import json
from pathlib import Path

from .adjudicate import adjudicate_chunk
from .alt_asr import run_alt_asr_on_chunk, alt_asr_runtime_info
from .audio import nemo_probe_has_text, split_wav_fixed, vad_likely_speech, vad_stats, wav_info, wav_rms
from .config import AppConfig
from .hypotheses import hypothesis_paths
from .ui import progress_desc
from .utils import clean_repetition, combine_files, has_words, is_usable_text_file, transcript_stats, words_per_minute
from .tasker_notify import send_with_tailscale

def chunk_paths(base: str, idx: int):
    transcripts_dir = Path("transcripts")
    transcripts_dir.mkdir(exist_ok=True)
    clean_dir = transcripts_dir / "clean"
    ts_dir = transcripts_dir / "timestamps"
    clean_dir.mkdir(parents=True, exist_ok=True)
    ts_dir.mkdir(parents=True, exist_ok=True)

    chunk_txt = clean_dir / f"{base}_{idx:03d}.txt"
    chunk_ts = ts_dir / f"{base}_{idx:03d}.timestamps.txt"

    combined_txt = transcripts_dir / f"{base}.txt"
    combined_ts = transcripts_dir / f"{base}.timestamps.txt"
    return chunk_txt, chunk_ts, combined_txt, combined_ts


def _run_recover_subchunks(
    model,
    chunk_path: Path,
    base: str,
    idx: int,
    *,
    tmp_dir: Path | None,
) -> tuple[str, str]:
    _tmp = tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks")
    try:
        subs = split_wav_fixed(chunk_path, _tmp / "_gpu_recover_subchunks" / f"{base}_{idx:03d}", segment_s=30)
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
        return " ".join(texts).strip(), ""
    except Exception as e:
        return "", str(e)


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

    # NOTE: We intentionally stage work to minimize GPU churn and prevent alt-ASR
    # from running on every chunk.
    #
    # Phase A: Parakeet baseline on *all* chunks.
    # Phase B: Parakeet subchunk recovery only for "needs recovery" chunks.
    # Phase C: Alt-ASR (Whisper) only for "needs recovery" chunks (initial chunks only).
    # Phase D: Post-ASR adjudication combining baseline/recover/alt_asr.
    clean_parts: list[Path] = []
    # Hypotheses to run conservative repetition cleanup on.
    # Start with alt_asr only (Whisper loop artifacts). Extend later if desired.
    DEDUP_HYPOTHESES = {"alt_asr", "recover", "baseline"}
    ts_parts: list[Path] = []
    total = len(chunks)
    suspicious_empty = 0
    empty_ok = 0
    recovered_low_coverage = 0

    # Indices where we should run recovery + alt-ASR.
    rerun_idxs: list[int] = []

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

        baseline_ok = False
        recover_ok = False
        alt_asr_ok = False
        nemo_error = ""
        recover_error = ""
        alt_asr_error = ""
        adjudicate_error = ""

        hpaths = hypothesis_paths(base, idx)
        hpaths["dir"].mkdir(parents=True, exist_ok=True)

        # Do not short-circuit per-chunk work when adjudication already exists.
        # We rebuild the combined transcript later, preferring adjudicated outputs.
        need_clean = not chunk_txt.exists()
        need_ts = timestamps and (not chunk_ts.exists())

        result_text = ""
        result_ts_words = []
        out = None
        try:
            try:
                out = model.transcribe([str(c)], timestamps=timestamps, verbose=False)
            except TypeError:
                out = model.transcribe([str(c)], timestamps=timestamps)
        except Exception as e:
            nemo_error = str(e)

        if out:
            result_text = (getattr(out[0], "text", "") or "")
            result_ts_words = (out[0].timestamp.get("word", []) if getattr(out[0], "timestamp", None) else [])
            if has_words(result_text):
                hpaths["baseline"].write_text(result_text.strip() + "\n", encoding="utf-8")

        baseline_ok = is_usable_text_file(hpaths["baseline"])

        # Compute WPM once (0 if we have no text).
        wc = len((result_text or "").strip().split())
        wpm = words_per_minute(wc, float(meta.get("duration_s", 0.0) or 0.0))

        # We only treat a chunk as "requires rerun" if:
        # - Parakeet errored, OR
        # - Parakeet produced empty/low-WPM AND a probe confirms there is transcribable speech.
        low_wpm_candidate = (wpm > 0.0 and wpm < 35.0)
        empty_candidate = (not has_words(result_text))

        # For rerun gating: VAD is a model-independent signal.
        # Only compute it for suspicious chunks (empty or low WPM), not for every chunk.
        vad_says_speech = False
        if (low_wpm_candidate or empty_candidate):
            try:
                vad_says_speech = bool(vad_likely_speech(c))
            except Exception:
                vad_says_speech = False

        probe_lowwpm_ok = False
        probe_empty_ok = False

        if not allow_fallback:
            _tmp = tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks")

            if low_wpm_candidate:
                try:
                    probe_lowwpm_ok = nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0)
                except Exception as e:
                    recover_error = recover_error or f"probe_failed: {e}"
                if probe_lowwpm_ok:
                    flags.append("low_wpm_probe_confirmed")

            if empty_candidate:
                try:
                    probe_empty_ok = nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0)
                except Exception as e:
                    recover_error = recover_error or f"probe_failed: {e}"
                if probe_empty_ok:
                    flags.append("empty_probe_confirmed")
                    suspicious_empty += 1
                else:
                    # Only count as "empty_ok" if the chunk is empty AND probe does not confirm speech.
                    empty_ok += 1

        # "Needs recovery" is defined as: Parakeet errored, or the NeMo probe confirms
        # there's transcribable speech despite low/empty output.
        # (We keep VAD for diagnostics, but we do not trigger alt-ASR on VAD alone.)
        rerun_needed = bool(nemo_error) or probe_lowwpm_ok or probe_empty_ok
        if rerun_needed:
            rerun_idxs.append(idx)

        # Phase A output: write baseline chunk text now. Later phases may overwrite.
        if need_clean:
            chunk_txt.write_text((result_text or "").strip() + "\n", encoding="utf-8")

        if timestamps and need_ts:
            chunk_ts.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{w['start']:.2f}s→{w['end']:.2f}: {w['word']}\n" for w in result_ts_words]
            chunk_ts.write_text("".join(lines), encoding="utf-8")
        if timestamps and chunk_ts.exists():
            ts_parts.append(chunk_ts)

        if audit_path is not None:
            vad = vad_stats(c, aggressiveness=3, frame_ms=30)
            vad_likely = vad_likely_speech(c)
            st = transcript_stats((result_text or "").strip())
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
                "baseline_ok": baseline_ok,
                "recover_ok": recover_ok,
                "alt_asr_ok": alt_asr_ok,
                "nemo_error": nemo_error,
                "recover_error": recover_error,
                "alt_asr_error": alt_asr_error,
                "adjudicate_error": adjudicate_error,
                "wpm": wpm,
                "vad": vad,
                "vad_likely_speech": vad_likely,
                "transcript": st,
                "hyp_baseline_path": str(hpaths["baseline"]) if baseline_ok else "",
                "hyp_recover_path": str(hpaths["recover"]) if recover_ok else "",
                "hyp_alt_asr_path": str(hpaths["alt_asr"]) if alt_asr_ok else "",
                "adjudicated_path": str(hpaths["adjudicated_txt"]) if used_adjudicate else "",
                "flags": flags,
            }
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if progress is not None and chunk_task_id is not None:
            progress.advance(chunk_task_id, 1)

    # ── Phase B: Parakeet subchunk recovery (needs recovery only) ─────────────
    if (not allow_fallback) and rerun_idxs:
        for idx in rerun_idxs:
            c = Path(chunks[idx])
            chunk_txt, _chunk_ts, _combined_txt, _combined_ts = chunk_paths(base, idx)
            hpaths = hypothesis_paths(base, idx)
            hpaths["dir"].mkdir(parents=True, exist_ok=True)

            if is_usable_text_file(hpaths["recover"]):
                continue

            recovered, recover_error = _run_recover_subchunks(model, c, base, idx, tmp_dir=tmp_dir)
            if has_words(recovered):
                hpaths["recover"].write_text(recovered.strip() + "\n", encoding="utf-8")
                # Prefer recover for chunk text pre-adjudication.
                try:
                    chunk_txt.write_text(recovered.strip() + "\n", encoding="utf-8")
                except Exception:
                    pass
                recovered_low_coverage += 1

            if audit_path is not None:
                rec_recover = {
                    "phase": "recover",
                    "base": base,
                    "chunk_index": idx,
                    "chunk_wav": str(c),
                    "recover_ok": bool(is_usable_text_file(hpaths["recover"])),
                    "recover_error": recover_error or "",
                    "hyp_recover_path": str(hpaths["recover"]) if is_usable_text_file(hpaths["recover"]) else "",
                }
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec_recover, ensure_ascii=False) + "\n")

    # ── Phase C: Alt-ASR (Whisper) on needs recovery chunks only ──────────────
    if cfg.alt_asr_enabled and rerun_idxs:
        for idx in rerun_idxs:
            c = Path(chunks[idx])
            chunk_txt, _chunk_ts, _combined_txt, _combined_ts = chunk_paths(base, idx)
            hpaths = hypothesis_paths(base, idx)
            hpaths["dir"].mkdir(parents=True, exist_ok=True)

            if is_usable_text_file(hpaths["alt_asr"]):
                continue

            alt_asr_error = ""
            try:
                alt_txt = run_alt_asr_on_chunk(cfg, c)
                if "alt_asr" in DEDUP_HYPOTHESES and has_words(alt_txt):
                    alt_txt = clean_repetition(alt_txt, min_ngram_words=12)
                if has_words(alt_txt):
                    hpaths["alt_asr"].write_text(alt_txt.strip() + "\n", encoding="utf-8")
                
                # Meta written after load so it reflects actual runtime.
                try:
                    hpaths["alt_asr_meta"].write_text(
                        json.dumps(alt_asr_runtime_info(), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                # If baseline was empty and we didn't recover, prefer alt-ASR for chunk text.
                if (not is_usable_text_file(hpaths["recover"])):
                    try:
                        cur = chunk_txt.read_text(encoding="utf-8", errors="ignore").strip() if chunk_txt.exists() else ""
                    except Exception:
                        cur = ""
                    if (not has_words(cur)) and has_words(alt_txt):
                        try:
                            chunk_txt.write_text(alt_txt.strip() + "\n", encoding="utf-8")
                        except Exception:
                            pass
            except Exception as e:
                alt_asr_error = str(e)
                try:
                    hpaths["alt_asr_error"].write_text(alt_asr_error.strip() + "\n", encoding="utf-8")
                except Exception:
                    pass

            if audit_path is not None:
                rec_alt = {
                    "phase": "alt_asr",
                    "base": base,
                    "chunk_index": idx,
                    "chunk_wav": str(c),
                    "alt_asr_ok": bool(is_usable_text_file(hpaths["alt_asr"])),
                    "alt_asr_error": alt_asr_error or "",
                    "hyp_alt_asr_path": str(hpaths["alt_asr"]) if is_usable_text_file(hpaths["alt_asr"]) else "",
                    "hyp_alt_asr_meta": str(hpaths["alt_asr_meta"]) if hpaths["alt_asr_meta"].exists() else "",
                }
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec_alt, ensure_ascii=False) + "\n")

    # Phase D: post-ASR adjudication (after Parakeet is released from VRAM).
    if cfg.adjudicate:
        # Best-effort release of GPU memory held by ASR model.
        try:
            model = None  # drop reference
        except Exception:
            pass
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Run adjudication only where we have >=2 usable hypotheses and no adjudicated output yet.
        for idx, c in enumerate(chunks):
            c = Path(c)
            hpaths = hypothesis_paths(base, idx)
            hpaths["dir"].mkdir(parents=True, exist_ok=True)

            # Skip if adjudication already done.
            if is_usable_text_file(hpaths["adjudicated_txt"]):
                continue

            # Read hypotheses from disk (resume-safe, no ASR reruns).
            h = {
                "baseline": hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip()
                if hpaths["baseline"].exists()
                else "",
                "recover": hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip()
                if hpaths["recover"].exists()
                else "",
                "alt_asr": hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip()
                if hpaths["alt_asr"].exists()
                else "",
            }

            # Conservative repetition cleanup (resume-safe).
            # This prevents a loopy hypothesis from dominating adjudication input.
            if "alt_asr" in DEDUP_HYPOTHESES and has_words(h.get("alt_asr", "")):
                h["alt_asr"] = clean_repetition(h["alt_asr"], min_ngram_words=12)
            if "recover" in DEDUP_HYPOTHESES and has_words(h.get("recover", "")):
                h["recover"] = clean_repetition(h["recover"], min_ngram_words=12)
            if "baseline" in DEDUP_HYPOTHESES and has_words(h.get("baseline", "")):
                h["baseline"] = clean_repetition(h["baseline"], min_ngram_words=12)

            # Need >= 2 hypotheses with words to adjudicate.
            if sum(1 for v in h.values() if has_words(v)) < 2:
                continue

            meta = wav_info(c)
            post_flags: list[str] = ["post_asr_adjudicate"]
            if is_usable_text_file(hpaths["recover"]):
                post_flags.append("has_recover_hyp")
            if is_usable_text_file(hpaths["alt_asr"]):
                post_flags.append("has_alt_asr_hyp")

            final_text, j, used, adj_err = adjudicate_chunk(
                cfg,
                h,
                chunk_seconds=float(meta.get("duration_s", 0.0) or 0.0),
                flags=post_flags,
            )

            if used and has_words(final_text):
                if isinstance(j, dict):
                    hpaths["adjudicated_json"].write_text(
                        json.dumps(j, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                hpaths["adjudicated_txt"].write_text(final_text.strip() + "\n", encoding="utf-8")

            # Optional: write a second audit record for adjudication phase (append-only).
            if audit_path is not None:
                rec2 = {
                    "phase": "adjudicate",
                    "base": base,
                    "chunk_index": idx,
                    "chunk_wav": str(c),
                    "used_adjudicate": bool(used and has_words(final_text)),
                    "adjudicate_error": adj_err or "",
                    "hyp_baseline_path": str(hpaths["baseline"]) if is_usable_text_file(hpaths["baseline"]) else "",
                    "hyp_recover_path": str(hpaths["recover"]) if is_usable_text_file(hpaths["recover"]) else "",
                    "hyp_alt_asr_path": str(hpaths["alt_asr"]) if is_usable_text_file(hpaths["alt_asr"]) else "",
                    "adjudicated_path": str(hpaths["adjudicated_txt"]) if is_usable_text_file(hpaths["adjudicated_txt"]) else "",
                    "flags": post_flags,
                }
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                with audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec2, ensure_ascii=False) + "\n")

        # Rebuild combined transcript preferring adjudicated outputs when present.
        final_parts: list[Path] = []
        for idx, _c in enumerate(chunks):
            chunk_txt, _chunk_ts, _combined_txt, _combined_ts = chunk_paths(base, idx)
            hpaths = hypothesis_paths(base, idx)
            if is_usable_text_file(hpaths["adjudicated_txt"]):
                final_parts.append(hpaths["adjudicated_txt"])
            elif is_usable_text_file(chunk_txt):
                final_parts.append(chunk_txt)
        clean_parts = final_parts

    # If adjudication is disabled, build the combined transcript from the (possibly
    # updated) per-chunk clean text files.
    if not cfg.adjudicate:
        clean_parts = []
        for idx, _c in enumerate(chunks):
            chunk_txt, _chunk_ts, _combined_txt, _combined_ts = chunk_paths(base, idx)
            if is_usable_text_file(chunk_txt):
                clean_parts.append(chunk_txt)

    combine_files(combined_txt, clean_parts)
    if timestamps:
        combine_files(combined_ts, ts_parts)

    if (not allow_fallback) and (suspicious_empty or empty_ok):
        print(f"⚠️  {base}: empty chunks (ok={empty_ok}, suspicious={suspicious_empty}), rescued_low_coverage={recovered_low_coverage}.")

    print(f"📄 Combined transcript: {combined_txt}")
    send_with_tailscale("Transcription Completed", "The latest transcript has been compiled!")
    return combined_txt, combined_ts
