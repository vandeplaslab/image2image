"""Rename package to represent the macOS platform + version."""

import inspect
from pathlib import Path

from image2image import __version__

parent = Path(inspect.getfile(lambda: None)).parent.parent.resolve()
print("Parent:", parent)
source_dir = parent / "dist" / "image2image.tar.gz"
assert source_dir.exists(), "Source directory does not exist"
output_dir = parent / "dist" / f"image2image-v{__version__}-linux_arm64.tar.gz"
source_dir.rename(output_dir)
