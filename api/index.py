"""Vercel entry point. Vercel's FastAPI support finds the ASGI `app` here and sends it every request
with its original path (no rewrites needed: a rewrite to /api/index would hide the real path)."""

import logging
import sys
from pathlib import Path

# Import the package from src/ whether or not the build installed it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from startsit.web.app import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

__all__ = ["app"]
