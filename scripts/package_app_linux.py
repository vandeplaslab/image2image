"""Zip file."""

import inspect
from pathlib import Path
from shutil import make_archive

from koyo.timer import MeasureTimer

from image2image import __version__

print("Started tarring folder")
with MeasureTimer() as timer:
    parent = Path(inspect.getfile(lambda: None)).parent.parent.resolve()
    print("Parent directory:", parent)
    source_dir = parent / "dist" / "image2image"
    assert source_dir.exists(), "Source directory does not exist"
    output_dir = parent / "dist" / f"image2image-v{__version__}-linux_amd64"
    make_archive(output_dir, "gztar", source_dir)
print(f"Finished tarring folder in {timer()}")
