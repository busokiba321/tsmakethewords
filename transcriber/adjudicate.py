from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from .config import AppConfig
from .utils import has_words, repetition_ratio


def build_prompt(hyps: dict[str, str], *, chunk_seconds: float, flags: list[str]) -> str:
    def norm(s: str) -> str:
        return (s or "").strip()

    a = norm(hyps.get("baseline", ""))
    b = norm(hyps.get("recover", ""))
    c = norm(hyps.get("alt_asr", ""))
    flag_line = ", ".join(flags) if flags else "none"
    return (
        "You are an ASR transcript adjudicator.\n"
        "Task: produce the best possible transcript for this audio chunk by selecting and combining ONLY words\n"
        "that appear in the provided hypotheses. You MUST be extractive-only.\n\n"
        "Rules:\n"
        "1) Do NOT add or invent any words that do not appear in at least one hypothesis.\n"
        "2) Prefer coherent, grammatical phrasing.\n"
        "3) Remove duplicated phrases and obvious repetition.\n"
        "4) If uncertain between two options, choose the version that is present verbatim in one hypothesis.\n"
        "5) If all hypotheses are bad, return the least-wrong one; never hallucinate.\n\n"
        f"Chunk duration seconds: {chunk_seconds:.2f}\n"
        f"Flags: {flag_line}\n\n"
        "Return STRICT JSON only with keys:\n"
        '{"final": "...", "used": ["baseline","recover","alt_asr"], "confidence": "high|medium|low", "notes": ["tag",...]}\n\n'
        "Hypothesis baseline:\n"
        f"{a}\n\n"
        "Hypothesis recover:\n"
        f"{b}\n\n"
        "Hypothesis alt_asr:\n"
        f"{c}\n"
    )


def run_llama_cpp_json(cfg: AppConfig, prompt: str) -> tuple[dict[str, Any] | None, str]:
    cmd = [
        cfg.llama_bin,
        "-m",
        cfg.llama_model,
        "-c",
        str(cfg.llama_ctx),
        "--seed",
        str(cfg.llama_seed),
        "--temp",
        str(cfg.llama_temp),
        "--top-p",
        str(cfg.llama_top_p),
        "-n",
        str(cfg.llama_max_tokens),
        "-t",
        str(cfg.llama_threads),
        "--gpu-layers",
        str(cfg.llama_gpu_layers),
        "-p",
        prompt,
    ]
    if cfg.llama_grammar:
        cmd.extend(["--grammar-file", cfg.llama_grammar])

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    except Exception as e:
        return None, f"llama invocation failed: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        return None, f"llama nonzero exit ({proc.returncode}): {err[:400]}"

    out = (proc.stdout or "").strip()
    if not out:
        return None, "llama produced empty stdout"
    m = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not m:
        return None, "llama output missing JSON object"
    try:
        return json.loads(m.group(0)), ""
    except Exception as e:
        return None, f"failed to parse llama JSON: {e}"


def choose_fallback_best(hyps: dict[str, str]) -> str:
    cands = []
    for k in ("recover", "alt_asr", "baseline"):
        t = (hyps.get(k) or "").strip()
        if not t:
            continue
        cands.append((repetition_ratio(t), len(t.split()), t))
    if not cands:
        return ""
    cands.sort(key=lambda x: (x[0], -x[1]))
    return cands[0][2]


def adjudicate_chunk(
    cfg: AppConfig,
    hyps: dict[str, str],
    *,
    chunk_seconds: float,
    flags: list[str],
) -> tuple[str, dict[str, Any] | None, bool, str]:
    multi = sum(1 for v in hyps.values() if has_words(v)) >= 2
    should = multi and (not cfg.adjudicate_only_when_flagged or bool(flags))
    if not (cfg.adjudicate and should):
        return (hyps.get("baseline") or "").strip(), None, False, ""

    prompt = build_prompt(hyps, chunk_seconds=chunk_seconds, flags=flags)
    j, adjudicate_error = run_llama_cpp_json(cfg, prompt)
    if isinstance(j, dict) and has_words(str(j.get("final", ""))):
        return str(j["final"]).strip(), j, True, ""

    return choose_fallback_best(hyps) or (hyps.get("baseline") or "").strip(), j if isinstance(j, dict) else None, False, adjudicate_error
