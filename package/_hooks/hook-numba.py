from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("numba")
hiddenimports = [
    "numba",
    "numba.core",
    "numba.core.old_boxing",
    "numba.core.types.old_scalars",
    "numba.core.datamodel.old_models",
    "numba.core.typing",
    "numba.core.typing.old_builtins",
    "numba.core.typing.old_mathdecl",
    "numba.core.typing.old_cmathdecl",
    "numba.core.typing.cffi_utils",
    "numba.cpython",
    "numba.cpython.old_builtins",
    "numba.cpython.old_hashing",
    "numba.cpython.old_mathimpl",
    "numba.cpython.old_numbers",
    "numba.cpython.old_tupleobj",
    "numba.experimental",
    "numba.experimental.jitclass",
    "numba.experimental.jitclass.base",
    "numba.experimental.jitclass.boxing",
    "numba.experimental.jitclass._box",
    "numba.np",
    "numba.np.old_arraymath",
    "numba.np.random",
    "numba.np.random.old_distributions",
    "numba.np.random.old_random_methods",

]
