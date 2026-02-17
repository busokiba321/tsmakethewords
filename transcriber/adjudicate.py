from __future__ import annotations

import base64
import json
import re
import atexit
import subprocess
import time
import socket
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError
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
        '{"final_lines": ["...","..."], "used": ["baseline","recover","alt_asr"], "confidence": "high|medium|low", "notes": ["tag",...]}\n\n'
        "IMPORTANT: Put the transcript into final_lines as an array of short strings (1–2 sentences per element). Do NOT return one huge string.\n"
        "Hypothesis baseline:\n"
        f"{a}\n\n"
        "Hypothesis recover:\n"
        f"{b}\n\n"
        "Hypothesis alt_asr:\n"
        f"{c}\n"
    )

_LLAMA_SERVER_PROC: subprocess.Popen[str] | None = None
_LLAMA_SERVER_URL: str | None = None
_GRAMMAR_CACHE: str | None = None


def _port_is_listening(host: str, port: int, timeout_s: float = 0.25) -> bool:
    """
    Returns True if something is listening on host:port (regardless of what it is).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False

def _find_free_port(host: str, start_port: int, tries: int = 50) -> int:
    """
    Scan upward from start_port until we find a port with nothing listening.
    """
    port = int(start_port)
    for _ in range(int(tries)):
        if not _port_is_listening(host, port):
            return port
        port += 1
    raise RuntimeError(f"Could not find a free port starting at {start_port} after {tries} tries")


def _read_grammar_text(cfg: AppConfig) -> str:
    global _GRAMMAR_CACHE
    if _GRAMMAR_CACHE is not None:
        return _GRAMMAR_CACHE
    if not getattr(cfg, "llama_grammar", ""):
        _GRAMMAR_CACHE = ""
        return _GRAMMAR_CACHE
    try:
        _GRAMMAR_CACHE = open(cfg.llama_grammar, "r", encoding="utf-8").read()
    except Exception:
        _GRAMMAR_CACHE = ""
    return _GRAMMAR_CACHE

def _ensure_llama_server(cfg: AppConfig) -> str:
    """
    Start llama-server once (persistent) and return base URL.
    Uses /health to wait for readiness. :contentReference[oaicite:2]{index=2}
    """
    global _LLAMA_SERVER_PROC, _LLAMA_SERVER_URL
    if _LLAMA_SERVER_URL is not None and _LLAMA_SERVER_PROC is not None and _LLAMA_SERVER_PROC.poll() is None:
        return _LLAMA_SERVER_URL

    # Infer llama-server next to llama-cli if possible
    server_bin = getattr(cfg, "llama_server_bin", None) or ""
    if not server_bin:
        if cfg.llama_bin.endswith("llama-cli"):
            server_bin = cfg.llama_bin[:-len("llama-cli")] + "llama-server"
        else:
            server_bin = "llama-server"

    host = getattr(cfg, "llama_server_host", "127.0.0.1")
    desired_port = int(getattr(cfg, "llama_server_port", 18080))

    # IMPORTANT: Do NOT reuse an externally launched server. It may not have grammar enforced,
    # which causes some chunks to return plain text instead of strict JSON.
    # If desired_port is already in use, start our own managed server on the next free port.
    port = desired_port
    if _port_is_listening(host, port):
        port = _find_free_port(host, port + 1)

    _LLAMA_SERVER_URL = f"http://{host}:{port}"


    cmd = [
        server_bin,
        "-m", cfg.llama_model,
        "-c", str(cfg.llama_ctx),
        "-t", str(cfg.llama_threads),
        "--seed", str(cfg.llama_seed),
        "--temp", str(cfg.llama_temp),
        "--top-p", str(cfg.llama_top_p),
        "--n-gpu-layers", str(cfg.llama_gpu_layers),
        "--host", host,
        "--port", str(port),
    ]

    # Enforce output schema with grammar at server startup (more reliable than per-request inline grammar).
    if getattr(cfg, "llama_grammar", ""):
        cmd.extend(["--grammar-file", cfg.llama_grammar])

    _LLAMA_SERVER_PROC = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def _cleanup() -> None:
        global _LLAMA_SERVER_PROC
        try:
            if _LLAMA_SERVER_PROC is not None and _LLAMA_SERVER_PROC.poll() is None:
                _LLAMA_SERVER_PROC.terminate()
        except Exception:
            pass

    atexit.register(_cleanup)

    # Wait for readiness via /health (returns 503 while loading) :contentReference[oaicite:3]{index=3}
    deadline = time.time() + float(getattr(cfg, "llama_server_startup_timeout_s", 300))
    while time.time() < deadline:
        if _LLAMA_SERVER_PROC.poll() is not None:
            raise RuntimeError("llama-server exited during startup")
        try:
            with urlrequest.urlopen(_LLAMA_SERVER_URL + "/health", timeout=2) as r:
                if r.status == 200:
                    return _LLAMA_SERVER_URL
        except Exception:
            pass
        time.sleep(1.0)

    raise TimeoutError("llama-server did not become ready in time")

def _run_llama_server_completion(cfg: AppConfig, prompt: str, n_predict: int) -> tuple[dict[str, Any] | None, str]:
    """
    POST /completion with grammar field. :contentReference[oaicite:4]{index=4}
    Returns (json_obj_or_none, raw_content_text)
    """
    base = _ensure_llama_server(cfg)
    body = {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "seed": int(cfg.llama_seed),
        "temperature": float(cfg.llama_temp),
        "top_p": float(cfg.llama_top_p),
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(
        base + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=float(getattr(cfg, "llama_timeout_s", 1000))) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    j = json.loads(raw)
    def _pick_content(obj: dict[str, Any]) -> str:
        # llama.cpp server has had a few response shapes over time
        # Try the common ones in order.
        v = obj.get("content")
        if isinstance(v, str) and v.strip():
            return v
        v = obj.get("completion")
        if isinstance(v, str) and v.strip():
            return v
        # OpenAI-like
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                v = c0.get("text") or c0.get("content")
                if isinstance(v, str) and v.strip():
                    return v
        # Sometimes server nests result
        res = obj.get("result")
        if isinstance(res, dict):
            v = res.get("content") or res.get("completion")
            if isinstance(v, str) and v.strip():
                return v
        return ""

    content = _pick_content(j).strip()
    return j, content

def _extract_first_json_object(text: str) -> str | None:
    """
    Best-effort: extract the first balanced JSON object from text.
    This lets us salvage valid JSON even if extra tokens or partial trailing output exists.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None

def _parse_json_strict_or_salvage(content: str) -> tuple[dict[str, Any] | None, str]:
    """
    Try strict json.loads(content). If that fails, salvage the first balanced JSON object.
    Returns (parsed_or_none, error_message_or_empty).
    """
    if not content:
        return None, "empty content"
    try:
        return json.loads(content), ""
    except Exception as e1:
        candidate = _extract_first_json_object(content)
        if not candidate:
            return None, f"non-JSON content: {e1}"
        try:
            return json.loads(candidate), ""
        except Exception as e2:
            return None, f"non-JSON content: {e2}"


def _run_llama_cli_json(cfg: AppConfig, prompt: str) -> tuple[dict[str, Any] | None, str]:
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
        str(getattr(cfg, "llama_max_tokens", 256)),
        "-t",
        str(cfg.llama_threads),
        "--n-gpu-layers",
        str(cfg.llama_gpu_layers),
        "-p",
        prompt,
    ]
    if cfg.llama_grammar:
        cmd.extend(["--grammar-file", cfg.llama_grammar])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=getattr(cfg, "llama_timeout_s", 1000),
        )
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

def run_llama_cpp_json(cfg: AppConfig, prompt: str) -> tuple[dict[str, Any] | None, str]:
    # Prefer persistent llama-server to avoid per-chunk model reload overhead (critical on Vulkan).
    if getattr(cfg, "llama_use_server", True):
        base_n = int(getattr(cfg, "llama_max_tokens", 256))
        max_retry_n = int(getattr(cfg, "llama_max_tokens_retry", 768))

        # Attempt 1
        try:
            _j, content = _run_llama_server_completion(cfg, prompt, n_predict=base_n)
        except (URLError, HTTPError, TimeoutError, RuntimeError) as e:
            return None, f"llama-server error: {e}"

        if not content:
            # Give a more useful error than "Expecting value"
            try:
                preview = json.dumps(_j)[:300]
            except Exception:
                preview = str(_j)[:300]
            return None, f"llama-server returned empty completion content; raw={preview}"

        parsed, perr = _parse_json_strict_or_salvage(content)
        if parsed is not None:
            return parsed, ""

        # Helpful debugging: show what the server actually returned (trimmed)
        content_preview = content[:300].replace("\n", "\\n")
        # Attach full content (base64) so caller can do a safe plaintext fallback per-chunk
        content_b64 = base64.b64encode(content.encode("utf-8", errors="ignore")).decode("ascii")
        perr = f"{perr}; content_preview={content_preview}; __content_b64__={content_b64}"

        # Attempt 2 (retry once with higher token budget; keeps seed/temp/top_p fixed)
        if max_retry_n > base_n:
            try:
                _j2, content2 = _run_llama_server_completion(cfg, prompt, n_predict=max_retry_n)
            except (URLError, HTTPError, TimeoutError, RuntimeError) as e:
                return None, f"llama-server error after retry: {e}"

            if not content2:
                try:
                    preview2 = json.dumps(_j2)[:300]
                except Exception:
                    preview2 = str(_j2)[:300]
                return None, f"llama-server returned empty completion content after retry; raw={preview2}"

            parsed2, perr2 = _parse_json_strict_or_salvage(content2)
            if parsed2 is not None:
                return parsed2, ""
            
            content2_preview = content2[:300].replace("\n", "\\n")
            content2_b64 = base64.b64encode(content2.encode("utf-8", errors="ignore")).decode("ascii")
            perr2 = f"{perr2}; content_preview={content2_preview}; __content_b64__={content2_b64}"
            return None, f"llama-server returned non-JSON content: {perr2}"

        return None, f"llama-server returned non-JSON content: {perr}"

    # Fallback: llama-cli (slow; reloads model per call)
    return _run_llama_cli_json(cfg, prompt)

def _tokenize_words(s: str) -> list[str]:
    # Lowercase word tokens; keep apostrophes inside words
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", (s or "").lower())

def _is_extractive_only(candidate: str, hyps: dict[str, str]) -> bool:
    """
    Strict extractive check: every word token in candidate must appear in at least one hypothesis.
    Conservative but safe.
    """
    cand_tokens = _tokenize_words(candidate)
    if not cand_tokens:
        return False
    allowed: set[str] = set()
    for txt in hyps.values():
        allowed.update(_tokenize_words(txt))
    return all(t in allowed for t in cand_tokens)

def _make_plaintext_adjudication(candidate: str, hyps: dict[str, str]) -> dict[str, Any]:
    used: list[str] = []
    cand_norm = (candidate or "").strip()
    if cand_norm and isinstance(hyps.get("baseline"), str) and cand_norm in hyps["baseline"]:
        used.append("baseline")
    if cand_norm and isinstance(hyps.get("recover"), str) and cand_norm in hyps["recover"]:
        if "recover" not in used:
            used.append("recover")
    if cand_norm and isinstance(hyps.get("alt_asr"), str) and cand_norm in hyps["alt_asr"]:
        if "alt_asr" not in used:
            used.append("alt_asr")
    if not used:
        used = ["baseline", "recover"]

    lines = [ln.strip() for ln in re.split(r"\n+", cand_norm) if ln.strip()]
    if not lines:
        lines = [cand_norm] if cand_norm else []

    return {
        "final_lines": lines,
        "used": used,
        "confidence": "medium",
        "notes": ["plaintext_extractive_fallback"],
    }

def _extract_content_b64(err: str) -> str:
    """
    Pull __content_b64__=<...> out of an error string, decode, and return plaintext.
    Returns "" if missing/invalid.
    """
    if not err:
        return ""
    m = re.search(r"__content_b64__=([A-Za-z0-9+/=]+)", err)
    if not m:
        return ""
    try:
        return base64.b64decode(m.group(1)).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""
        
def _sanitize_plaintext_candidate(s: str) -> str:
    """
    llama-server sometimes returns plaintext transcript PLUS extra sections
    (e.g., 'Notes:', echoed instructions, or JSON-ish content).
    For our strict extractive fallback, keep only the leading transcript-like
    portion and cut at common section headers / JSON.
    """
    if not s:
        return ""
    text = s.strip()

    # Cut at common "non-transcript" section starters (case-insensitive).
    # We intentionally include variants we've seen in your logs.
    cut_markers = [
        r"\n\s*notes\s*:",                 # Notes:
        r"\n\s*this is a post-asr",        # This is a post-ASR...
        r"\n\s*task\s*:",                  # Task:
        r"\n\s*rules\s*:",                 # Rules:
        r"\n\s*return\s+strict\s+json",    # Return STRICT JSON...
        r"\n\s*hypothesis\s+baseline",     # Hypothesis baseline:
        r"\n\s*\{",                        # JSON starts
    ]
    for pat in cut_markers:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            text = text[: m.start()].strip()
            break

    # Also remove any trailing "Notes:" lines if they slipped through without a leading newline.
    # (e.g., ".... P right back.\n\nNotes: ...")
    text = re.split(r"\bnotes\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    return text

def _tokenize_words(s: str) -> list[str]:
    # Lowercase word tokens; keep apostrophes inside words
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", (s or "").lower())

def _is_extractive_only(candidate: str, hyps: dict[str, str]) -> bool:
    """
    Strict extractive check: every word token in candidate must appear in at least one hypothesis.
    This is conservative (may reject some valid punctuation/formatting differences), but safe.
    """
    cand_tokens = _tokenize_words(candidate)
    if not cand_tokens:
        return False
    allowed: set[str] = set()
    for txt in hyps.values():
        allowed.update(_tokenize_words(txt))
    return all(t in allowed for t in cand_tokens)

def _make_plaintext_adjudication(candidate: str, hyps: dict[str, str]) -> dict[str, Any]:
    """
    Convert plaintext candidate into our adjudication JSON schema.
    We keep it simple and deterministic.
    """
    # Determine "used" conservatively: which hypothesis contains the candidate verbatim (if any)
    used: list[str] = []
    cand_norm = (candidate or "").strip()
    if cand_norm and isinstance(hyps.get("baseline"), str) and cand_norm in hyps["baseline"]:
        used.append("baseline")
    if cand_norm and isinstance(hyps.get("recover"), str) and cand_norm in hyps["recover"]:
        if "recover" not in used:
            used.append("recover")
    if cand_norm and isinstance(hyps.get("alt_asr"), str) and cand_norm in hyps["alt_asr"]:
        if "alt_asr" not in used:
            used.append("alt_asr")
    if not used:
        # Unknown origin, but still extractive; record as ambiguous
        used = ["baseline", "recover"]

    # Split into short-ish lines to avoid giant JSON strings
    lines = [ln.strip() for ln in re.split(r"\n+", cand_norm) if ln.strip()]
    if not lines:
        lines = [cand_norm] if cand_norm else []

    return {
        "final_lines": lines,
        "used": used,
        "confidence": "medium",
        "notes": ["plaintext_extractive_fallback"],
    }

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

    # If llama-server ignored grammar and returned plaintext, accept ONLY if strictly extractive.
    # Use per-call embedded content (thread-safe) rather than globals.
    if not isinstance(j, dict) and adjudicate_error:
        raw = _extract_content_b64(adjudicate_error)
        candidate = _sanitize_plaintext_candidate(raw)
        if candidate and has_words(candidate) and _is_extractive_only(candidate, hyps):
            j = _make_plaintext_adjudication(candidate, hyps)
            adjudicate_error = ""

    # If parse failed or server returned garbage, keep going to fallback.
    final = ""
    if isinstance(j, dict) and isinstance(j.get("final_lines"), list):
        # Join lines safely; strip each; drop empties
        parts = []
        for x in j.get("final_lines", []):
            if isinstance(x, str):
                s = x.strip()
                if s:
                    parts.append(s)
        final = "\n".join(parts).strip()
    elif isinstance(j, dict) and isinstance(j.get("final"), str):
        # Back-compat if model returns old schema
        final = j["final"].strip()

    # Success path: return adjudicated text if it has words.
    if has_words(final):
        return final, j, True, ""

    # Failure path: keep hypotheses; return error so audit logs show why.
    return (
        choose_fallback_best(hyps) or (hyps.get("baseline") or "").strip(),
        j if isinstance(j, dict) else None,
        False,
        adjudicate_error,
    )