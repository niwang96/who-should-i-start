"""Vercel entry point: Vercel serves the ASGI `app` in this file; vercel.json routes every path here."""

import logging
import sys
from pathlib import Path

# Import the package from src/ whether or not the build installed it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from startsit.web.app import app

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

__all__ = ["app"]
