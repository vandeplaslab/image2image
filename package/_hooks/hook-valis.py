"""Valis hook."""
from koyo.utilities import is_installed
from PyInstaller.utils.hooks import collect_data_files

HAS_VALIS = is_installed("valis") and is_installed("pyvips")

if HAS_VALIS:
    datas = [] + collect_data_files("valis") + collect_data_files("pyvips") + collect_data_files("jpype")
    hiddenimports = ["valis", "pyvips", "jpype"]
