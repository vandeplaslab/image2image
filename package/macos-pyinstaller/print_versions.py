"""Print versions."""
from qtextra import __version__ as qtextra_version
from koyo import __version__ as koyo_version
from napari import __version__ as napari_version
from image2image import __version__ as image2image_version
from image2image_io import __version__ as image2image_io_version
from pydantic import __version__ as pydantic_version

print(f"qtextra: {qtextra_version}")
print(f"koyo: {koyo_version}")
print(f"napari: {napari_version}")
print(f"image2image: {image2image_version}")
print(f"image2image_io: {image2image_io_version}")
print(f"pydantic: {pydantic_version}")

