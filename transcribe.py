#!/usr/bin/env python
"""Compatibility entrypoint.

Transcriber was originally implemented as a single large script. It has been
refactored into a package under ./transcriber for maintainability.

Run:
  python transcribe.py <files>

Or:
  python -m transcriber.cli <files>
"""

from transcriber.cli import main


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("\n⛔ Closing (Ctrl+C).")
		raise SystemExit(130)