"""Print versions."""

import importlib

# Own modules
print("----- Own packages -----")
for module in [
    "image2image",
    "image2image_io",
    "image2image_reg",
    "koyo",
    "pydantic",
    "qtextra",
    "qtextraplot",
]:
    # import module and get version
    mod = importlib.import_module(module)
    version = mod.__version__
    print(f"{module}: {version}")


# Installed modules
print("----- Installed packages -----")
for module in [
    "SimpleITK",
    "itk",
    "napari",
    "numpy",
    "pandas",
]:
    # import module and get version
    mod = importlib.import_module(module)
    version = mod.__version__
    print(f"{module}: {version}")
