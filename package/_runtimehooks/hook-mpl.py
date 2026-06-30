"""Matplotlib hook to avoid font cache issues."""

import os
import sys
import tempfile

if getattr(sys, "frozen", False):
    cache_dir = os.path.join(tempfile.gettempdir(), "mpl_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
