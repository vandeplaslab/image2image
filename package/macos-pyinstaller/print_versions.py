"""Print versions."""

import importlib

for module in [
    "image2image",
    "image2image_io",
    "koyo",
    "napari",
    "numpy",
    "pandas",
    "pydantic",
    "qtextra",
    "qtextraplot",
]:
    # import module and get version
    mod = importlib.import_module(module)
    version = mod.__version__
    print(f"{module}: {version}")
