from __future__ import annotations

import json
from pathlib import Path

from .audio import wav_info, wav_rms, vad_stats, vad_likely_speech, nemo_probe_has_text, split_wav_fixed
from .config import AppConfig
from .hypotheses import hypothesis_paths
from .adjudicate import adjudicate_chunk
from .utils import has_words, transcript_stats, words_per_minute, is_usable_text_file, combine_files
from .ui import progress_desc


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


def transcribe_chunks(base, chunks, model, timestamps, cfg: AppConfig, progress=None, chunk_task_id=None, *, eou_mode: bool = False, tmp_dir: Path | None = None, allow_fallback: bool = True, audit_path: Path | None = None):
	transcripts_dir = Path("transcripts")
	transcripts_dir.mkdir(exist_ok=True)

	clean_parts: list[Path] = []
	ts_parts: list[Path] = []
	total = len(chunks)
	suspicious_empty = 0
	empty_ok = 0
	recovered_low_coverage = 0

	hyp_baseline: dict[str, str] = {}
	hyp_recover: dict[str, str] = {}
	hyp_alt_asr: dict[str, str] = {}

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

		if cfg.adjudicate and hpaths["adjudicated_txt"].exists() and is_usable_text_file(hpaths["adjudicated_txt"]):
			clean_parts.append(hpaths["adjudicated_txt"])
			if progress is not None and chunk_task_id is not None:
				progress.advance(chunk_task_id, 1)
			continue

		need_clean = not is_usable_text_file(chunk_txt)
		need_ts = timestamps and (not chunk_ts.exists())

		utt_id = f"{base}_{idx}"

		try:
			out = model.transcribe([str(c)], timestamps=timestamps, verbose=False)
		except Exception:
			out = None

		result_text = ""
		result_ts_words = []
		if out:
			r = out[0]
			result_text = (getattr(r, "text", "") or "")
			result_ts_words = (r.timestamp.get("word", []) if getattr(r, "timestamp", None) else [])

		if has_words(result_text):
			hyp_baseline[utt_id] = result_text.strip()
			hpaths["baseline"].write_text(result_text.strip() + "\n", encoding="utf-8")

		# GPU strict rescue (subchunks only)
		if not allow_fallback:
			wc = len((result_text or "").strip().split())
			wpm = words_per_minute(wc, float(meta.get("duration_s", 0.0) or 0.0))
			if wpm > 0.0 and wpm < 35.0:
				_tmp = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks"))
				if nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0):
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

		if (not allow_fallback) and (not has_words(result_text)):
			_tmp = (tmp_dir if tmp_dir is not None else Path(".tmp_audio_chunks"))
			has_text = nemo_probe_has_text(model, c, _tmp, probe_dur_s=15.0)
			if has_text:
				flags.append("empty_probe_confirmed")
				suspicious_empty += 1
			else:
				empty_ok += 1

		final_text = (result_text or "").strip()
		if cfg.adjudicate:
			h = {
				"baseline": hyp_baseline.get(utt_id, "") or (hpaths["baseline"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["baseline"].exists() else ""),
				"recover": hyp_recover.get(utt_id, "") or (hpaths["recover"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["recover"].exists() else ""),
				"alt_asr": hyp_alt_asr.get(utt_id, "") or (hpaths["alt_asr"].read_text(encoding="utf-8", errors="ignore").strip() if hpaths["alt_asr"].exists() else ""),
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
			try:
				chunk_ts.write_text(json.dumps(result_ts_words, ensure_ascii=False) + "\n", encoding="utf-8")
			except Exception:
				pass
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
				"flags": flags,
			}
			audit_path.parent.mkdir(parents=True, exist_ok=True)
			with audit_path.open("a", encoding="utf-8") as f:
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")

		if progress is not None and chunk_task_id is not None:
			progress.advance(chunk_task_id, 1)

	combined_txt = Path("transcripts") / f"{base}.txt"
	combined_ts = Path("transcripts") / f"{base}_timestamps.txt"
	combine_files(combined_txt, clean_parts)
	if timestamps:
		combine_files(combined_ts, ts_parts)

	if (not allow_fallback) and (suspicious_empty or empty_ok):
		print(f"⚠️  {base}: empty chunks (ok={empty_ok}, suspicious={suspicious_empty}), rescued_low_coverage={recovered_low_coverage}.")

	print(f"📄 Combined transcript: {combined_txt}")
	return combined_txt, combined_ts