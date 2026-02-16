from __future__ import annotations

from pathlib import Path


def hypothesis_dir(base: str) -> Path:
	return Path("transcripts") / "hypotheses" / base


def hypothesis_paths(base: str, idx: int) -> dict[str, Path]:
	out_dir = hypothesis_dir(base)
	stem = f"{base}_{idx:03d}"
	return {
		"dir": out_dir,
		"baseline": out_dir / f"{stem}.baseline.txt",
		"recover": out_dir / f"{stem}.recover.txt",
		"alt_asr": out_dir / f"{stem}.alt_asr.txt",
		"adjudicated_json": out_dir / f"{stem}.adjudicated.json",
		"adjudicated_txt": out_dir / f"{stem}.adjudicated.txt",
	}