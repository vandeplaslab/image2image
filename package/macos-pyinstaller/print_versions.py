"""Print versions."""
import importlib

for module in [
    "image2image",
    "image2image_io",
    "koyo",
    "napari",
    "pydantic",
    "qtextra",
]:
    # import module and get version
    mod = importlib.import_module(module)
    version = mod.__version__
    print(f"{module}: {version}")
