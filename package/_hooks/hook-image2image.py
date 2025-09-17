from PyInstaller.utils.hooks import collect_data_files
import image2image

hiddenimports = ["image2image", "image2image_io", "image2image_reg"]
datas = collect_data_files("image2image")
