from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("numba")
hiddenimports = [
    "numba",
    "numba.core",
    "numba.experimental",
    "numba.experimental.jitclass",
    "numba.experimental.jitclass.base",
    "numba.experimental.jitclass.boxing",
    "numba.experimental.jitclass._box",
    "numba.core.typing.cffi_utils",
]
