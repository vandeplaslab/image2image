import qtextra
import qtextraplot
    
from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ["qtextra", "qtextraplot"]
datas = collect_data_files("qtextra") + collect_data_files("qtextraplot")
