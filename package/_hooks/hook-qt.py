from qtpy import PYQT5, PYQT6, PYSIDE2, PYSIDE6

hiddenimports = [
    "qtpy",
    "magicgui.backends._qtpy",
]
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
