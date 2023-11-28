from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("xsdata_pydantic_basemodel")
hiddenimports = collect_submodules("xsdata_pydantic_basemodel")