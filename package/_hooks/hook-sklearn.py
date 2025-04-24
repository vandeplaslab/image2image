from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("sklearn")
hiddenimports = [
    "sklearn.neighbors._partition_nodes",
    "sklearn.utils._cython_blas",
    "sklearn.utils._estimator_html_repr",
]
