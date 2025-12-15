import qtextra  # noqa: F401
import qtextraplot  # noqa: F401
from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ["qtextra", "qtextraplot"]
datas = collect_data_files("qtextra") + collect_data_files("qtextraplot")
