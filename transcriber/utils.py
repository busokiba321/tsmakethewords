from __future__ import annotations

import re
from collections import Counter
from pathlib import Path


def has_words(text: str) -> bool:
    return bool(re.search(r"\w", (text or "").strip()))


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


def combine_files(out_path: Path, parts: list[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as outf:
        for p in parts:
            if not p.exists():
                continue
            outf.write(p.read_text(encoding="utf-8", errors="ignore").rstrip())
            outf.write("\n")
