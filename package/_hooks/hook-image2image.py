from PyInstaller.utils.hooks import collect_data_files
import ionglow

datas = collect_data_files("image2image")
hiddenimports = ["image2image"]
