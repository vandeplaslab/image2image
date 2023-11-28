"""PyInstaller setup script."""
import os
from pathlib import Path
import time
from image2image.assets import ICON_ICO
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, TOC, MERGE
from PyInstaller.utils.hooks import (
    get_package_paths,
    remove_prefix,
    PY_IGNORE_EXTENSIONS,
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    collect_all,
    copy_metadata,
)
import qtpy
import napari
import image2image
import debugpy._vendored
import imagecodecs


time_start = time.time()
block_cipher = None


def collect_pkg_data(package, include_py_files=False, subdir=None):
    """Collect package data."""
    # Accept only strings as packages.
    if type(package) is not str:
        raise ValueError

    pkg_base, pkg_dir = get_package_paths(package)
    if subdir:
        pkg_dir = os.path.join(pkg_dir, subdir)
    # Walk through all file in the given package, looking for data files.
    data_toc = TOC()
    for dir_path, dir_names, files in os.walk(pkg_dir):
        for f in files:
            extension = os.path.splitext(f)[1]
            if include_py_files or (extension not in PY_IGNORE_EXTENSIONS):
                source_file = os.path.join(dir_path, f)
                dest_folder = remove_prefix(dir_path, os.path.dirname(pkg_base) + os.sep)
                dest_file = os.path.join(dest_folder, f)
                data_toc.append((dest_file, source_file, "DATA"))
    return data_toc


def _make_analysis(path: str):
    return Analysis(
        [path],
        binaries=[],
        datas=[]
        # + collect_data_files("numba")
        + collect_data_files("qtextra")
        + collect_data_files("napari")
        + collect_data_files("xmlschema")
        + collect_data_files("ome_types")
        + collect_data_files("distributed")
        # + collect_data_files("imagecodecs")
        + collect_data_files("imzy")
        + collect_data_files("vispy")
        + collect_data_files("napari")
        + collect_data_files("image2image")
        + collect_data_files("freetype")
        + collect_data_files("xmlschema")
        # + collect_data_files("SimpleITK")
        # + collect_data_files("itk", include_py_files=True)
        + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=[]
        + [
            # scipy
            #               "scipy",
            #               "scipy.sparse.csgraph._validation",
            #               "scipy.linalg.cython_blas",
            #               "scipy.integrate",
            #               "scipy.special",
            #               "scipy.special._ufuncs_cxx",
            #               "scipy.special._ufuncs",
            #               "scipy.stats",
            #               "scipy._lib.messagestream",
            # numba - these actually don't work and the Jitclass must be disabled manually in numba
            # "numba",
            # "numba.core",
            # "numba.experimental",
            # "numba.experimental.jitclass",
            # "numba.experimental.jitclass.base",
            # "numba.experimental.jitclass.boxing",
            # "numba.experimental.jitclass._box",
            # "numba.core.typing.cffi_utils",
            # sklearn
            # "sklearn.neighbors._partition_nodes",
            # "sklearn.utils._cython_blas",
            # itk
            # "SimpleITK",
            # "itk",
            # "itk.Configuration",
            # other
            # "imagecodecs._shared",
            "pkg_resources",
            "six",
            "psygnal",
            "psygnal._signal",
            "pyside2",
            "qtpy",
            "vispy.app.backends._pyside2",
            "freetype",
            "magicgui.backends._qtpy",
            "imzy",
        ],
        # + [f"imagecodecs.{y}" for y in (x if x[0] == "_" else f"_{x}" for x in imagecodecs._extensions())],
        hookspath=[
            "hooks",
        ],
        runtime_hooks=[
            "runtimehooks/hook-bundle.py",
            "runtimehooks/hook-multiprocessing.py",
        ],
        excludes=[] + ["tcl", "Tkconstants", "Tkinter"],
        cipher=block_cipher,
    )


def _make_exe(pyz: PYZ, analysis: Analysis, name: str):
    """Make the executable."""
    return EXE(
        pyz,
        analysis.scripts,
        exclude_binaries=True,
        name=name,
        debug="all",
        strip=False,
        upx=True,
        console=True,
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
    )


# main app / launcher
launcher_analysis = _make_analysis("../../src/image2image/__main__.py")
launcher_exe = _make_exe(PYZ(launcher_analysis.pure), launcher_analysis, "image2image")
# # viewer app
# viewer_analysis = _make_analysis("../../src/image2image/__main_viewer__.py")
# viewer_exe = _make_exe(PYZ(viewer_analysis.pure), viewer_analysis, "image2viewer")
# # register app
# register_analysis = _make_analysis("../../src/image2image/__main_register__.py")
# register_exe = _make_exe(PYZ(register_analysis.pure), register_analysis, "image2register")
# # crop app
# crop_analysis = _make_analysis("../../src/image2image/__main_crop__.py")
# crop_exe = _make_exe(PYZ(crop_analysis.pure), crop_analysis, "image2crop")
# # convert app
# convert_analysis = _make_analysis("../../src/image2image/__main_convert__.py")
# convert_exe = _make_exe(PYZ(convert_analysis.pure), convert_analysis, "czi2tiff")
# # fusion app
# fusion_analysis = _make_analysis("../../src/image2image/__main_fusion.py")
# fusion_exe = _make_exe(PYZ(fusion_analysis.pure), fusion_analysis, "image2fusion")

# collect all
image2image_coll = COLLECT(
    # launcher
    launcher_exe,
    launcher_analysis.binaries,
    launcher_analysis.zipfiles,
    launcher_analysis.datas,
    # # viewer
    # viewer_exe,
    # viewer_analysis.binaries,
    # viewer_analysis.zipfiles,
    # viewer_analysis.datas,
    # # register
    # register_exe,
    # register_analysis.binaries,
    # register_analysis.zipfiles,
    # register_analysis.datas,
    # # crop
    # crop_exe,
    # crop_analysis.binaries,
    # crop_analysis.zipfiles,
    # crop_analysis.datas,
    # # convert
    # convert_exe,
    # convert_analysis.binaries,
    # convert_analysis.zipfiles,
    # convert_analysis.datas,
    # # fusion
    # fusion_exe,
    # fusion_analysis.binaries,
    # fusion_analysis.zipfiles,
    # fusion_analysis.datas,
    strip=False,
    debug="all",
    upx=True,
    name="image2image",
)

# Give information about build time
time_end = time.time()
print("Build image2image in {:.2f} seconds\n".format(time_end - time_start))
