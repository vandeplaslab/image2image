"""Print versions."""

import importlib

from koyo.timer import MeasureTimer

# Own modules
print("----- Own packages -----")
for module in [
    "image2image",
    "image2image_io",
    "image2image_reg",
    "koyo",
    "qtextra",
    "qtextraplot",
]:
    # import module and get version
    with MeasureTimer() as timer:
        mod = importlib.import_module(module)
        version = mod.__version__
        print(f"{module}: {version} ({timer(since_last=True)})")


# Installed modules
print("----- Installed packages -----")
for module in [
    "pydantic",
    "SimpleITK",
    "itk",
    "napari",
    "numpy",
    "pandas",
    "zarr",
    "imageio",
    "imagecodecs",
    "tifffile",
    "czifile",
]:
    # import module and get version
    with MeasureTimer() as timer:
        mod = importlib.import_module(module)
        version = mod.__version__
        print(f"{module}: {version} ({timer(since_last=True)})")
