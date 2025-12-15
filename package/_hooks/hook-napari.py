import imagecodecs  # noqa: F401
import napari  # noqa: F401
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("napari")
