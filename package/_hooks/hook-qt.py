from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from qtpy import PYQT5, PYQT6, PYSIDE2, PYSIDE6

hiddenimports = []
if PYQT5:
    hiddenimports += [
        "pyqt5",
    ]
elif PYQT6:
    hiddenimports += [
        "pyqt6",
    ]
elif PYSIDE2:
    hiddenimports += [
        "pyside2",
    ]
elif PYSIDE6:
    hiddenimports += [
        "pyside6",
    ]
