from PyInstaller.utils.hooks import collect_data_files, copy_metadata
import image2image

hiddenimports = ["image2image", "image2image_io", "image2image_reg"]
datas = collect_data_files("image2image")

# Collect metadata so that the backend can be discovered via `toga.backends` entry-point.
datas += copy_metadata("image2image")
datas += copy_metadata("image2image_io")
datas += copy_metadata("image2image_reg")
