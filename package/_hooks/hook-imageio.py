from PyInstaller.utils.hooks import collect_data_files, copy_metadata

import image2image  # noqa: F401

hiddenimports = ["imageio"]
# Collect data for the image2image package.
datas = collect_data_files("imageio")
# Collect metadata so that the backend can be discovered via `toga.backends` entry-point.
datas += copy_metadata("imageio")