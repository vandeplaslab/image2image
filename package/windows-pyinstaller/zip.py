from pathlib import Path
import inspect

from image2image import __version__
from koyo.timer import MeasureTimer
from shutil import make_archive

print("Started zipping folder")
with MeasureTimer() as timer:
    parent = Path(inspect.getfile(lambda: None)).parent.resolve()
    source_dir = parent / "dist" / "image2image"
    output_dir = parent / "dist" / f"image2image-v{__version__}-win_amd64.zip"
    make_archive(output_dir, "zip", source_dir)
print(f"Finished zipping folder in {timer()}")
