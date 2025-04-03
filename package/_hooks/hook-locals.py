from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("qtextra") + collect_data_files("qtextraplot") + collect_data_files("image2image")
