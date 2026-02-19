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

def _collapse_consecutive_duplicate_lines(text: str) -> str:
    """Remove consecutive identical lines (strip-compared). Preserves newlines."""
    lines = (text or "").splitlines()
    out: list[str] = []
    prev_key: str | None = None
    for line in lines:
        key = line.strip()
        if prev_key is not None and key and key == prev_key:
            continue
        out.append(line)
        prev_key = key
    return "\n".join(out)


def _collapse_adjacent_repeated_word_spans_in_line(line: str, *, min_words: int = 12) -> str:
    """
    Collapse immediate repeated spans of >= min_words words *within a single line*.
    Example: "a b c ... (12w) a b c ... (12w)" -> keep one copy.
    This is intentionally conservative: only adjacent repeats of equal-length spans.
    """
    words = (line or "").split()
    n = int(min_words)
    if n <= 0 or len(words) < (2 * n):
        return line

    out: list[str] = []
    i = 0
    L = len(words)
    while i < L:
        # If we have at least two consecutive n-word windows and they match, emit one and skip repeats.
        if i + (2 * n) <= L and words[i : i + n] == words[i + n : i + (2 * n)]:
            out.extend(words[i : i + n])
            j = i + n
            # Skip any further immediate repeats of the same n-gram.
            while j + n <= L and words[j : j + n] == words[i : i + n]:
                j += n
            i = j
            continue

        out.append(words[i])
        i += 1

    return " ".join(out)


def clean_repetition(text: str, *, min_ngram_words: int = 12) -> str:
    """
    Conservative repetition cleanup:
      1) Collapse consecutive identical lines (strip-compared)
      2) Within each line, collapse immediate repeated spans of >=min_ngram_words words

    This is designed to remove Whisper loop artifacts without changing content wording.
    """
    t = (text or "").strip()
    if not t:
        return ""

    t = _collapse_consecutive_duplicate_lines(t)
    lines = t.splitlines()
    cleaned = [
        _collapse_adjacent_repeated_word_spans_in_line(ln, min_words=min_ngram_words)
        for ln in lines
    ]
    return "\n".join(cleaned).strip()