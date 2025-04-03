"""PyInstaller setup script."""

import os
from pathlib import Path

import imagecodecs
import napari
import qtpy
import debugpy._vendored
from koyo.timer import MeasureTimer
from PyInstaller.building.build_main import COLLECT, EXE, MERGE, PYZ, TOC, Analysis
from PyInstaller.utils.hooks import (
    PY_IGNORE_EXTENSIONS,
    collect_data_files,
    get_package_paths,
    remove_prefix,
)

import image2image
from image2image.assets import ICON_ICO as ICON_APP_ICO
# TODO change to it's own icon
from image2image.assets import ICON_ICO as ICON_REG_ICO

block_cipher = None
# allowed values: all, imports, bootloader, noarchive
DEBUG_MODE = os.getenv("PYINSTALLER_DEBUG", False)
# allowed values: debug, info, warning, error, critical
LOG_MODE = os.getenv("PYINSTALLER_LOG", "DEBUG")
# allowed values: hide-early, minimize-late, minimize-early, hide-late
CONSOLE_MODE = os.getenv("PYINSTALLER_CONSOLE", "hide-early")

# allowed values: true, false
BUILD_REG = os.getenv("IMAGE2IMAGE_BUILD_REG", "true")

FILE_DIR = Path.cwd()
BASE_DIR = FILE_DIR.parent.parent
GITHUB_DIR = BASE_DIR.parent


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
    for dir_path, _dir_names, files in os.walk(pkg_dir):
        for f in files:
            extension = os.path.splitext(f)[1]
            if include_py_files or (extension not in PY_IGNORE_EXTENSIONS):
                source_file = os.path.join(dir_path, f)
                dest_folder = remove_prefix(dir_path, os.path.dirname(pkg_base) + os.sep)
                dest_file = os.path.join(dest_folder, f)
                data_toc.append((dest_file, source_file, "DATA"))
    return data_toc


def _make_analysis(path: str):
    print(f"Make analysis for {path}")
    return Analysis(
        [path],
        binaries=[],
        datas=[] +     collect_data_files("qtextra")
    + collect_data_files("qtextraplot")
    + collect_data_files("image2image")
    +     collect_data_files("napari")
    + collect_data_files("xmlschema")
    + collect_data_files("ome_types")
    + collect_data_files("distributed")
    + collect_data_files("freetype")
    + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=["freetype", "six", "pkg_resources"],
        hookspath=[
            "../_hooks",
        ],
        runtime_hooks=[
            "../_runtimehooks/hook-bundle.py",
            "../_runtimehooks/hook-multiprocessing.py",
        ],
        excludes=["tcl", "Tkconstants", "Tkinter"],
        cipher=block_cipher,
    )


def _make_exe(pyz: PYZ, analysis: Analysis, name: str, icon: Path = ICON_APP_ICO):
    """Make the executable."""
    return EXE(
        pyz,
        analysis.scripts,
        exclude_binaries=True,
        name=name,
        debug=DEBUG_MODE,
        strip=False,
        upx=True,
        console=True,  # False,
        hide_console=CONSOLE_MODE,
        bootloader_ignore_signals=False,
        icon=icon,
    )


# main app / launcher
with MeasureTimer() as timer:
    # registration
    extra_args = ()
    if BUILD_REG == "true":
        reg_analysis = _make_analysis(str(GITHUB_DIR / "image2image-reg" / "src" / "image2image_reg" / "__main__.py"))
        print(f"Analysis (reg) took {timer.format(timer.elapsed_since_last())}")
        reg_pyz = PYZ(reg_analysis.pure)
        print(f"PYZ (reg) took {timer.format(timer.elapsed_since_last())}")
        reg_exe = _make_exe(reg_pyz, reg_analysis, "i2reg", icon=ICON_REG_ICO)
        print(f"EXE (reg) took {timer.format(timer.elapsed_since_last())}")
        extra_args = (
            reg_exe,
            reg_analysis.binaries,
            reg_analysis.zipfiles,
            reg_analysis.datas,
        )

    # app
    launcher_analysis = _make_analysis(str(BASE_DIR / "src" / "image2image" / "__main__.py"))
    print(f"Analysis (app) took {timer.format(timer.elapsed_since_last())}")
    launcher_pyz = PYZ(launcher_analysis.pure)
    print(f"PYZ (app) took {timer.format(timer.elapsed_since_last())}")
    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image", icon=ICON_APP_ICO)
    print(f"EXE (app) took {timer.format(timer.elapsed_since_last())}")

    # collect all
    image2image_coll = COLLECT(
        # reg
        *extra_args,
        # launcher
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.zipfiles,
        launcher_analysis.datas,
        # other options
        strip=False,
        debug=DEBUG_MODE,
        upx=True,
        name="image2image",
    )
    print(f"COLLECT (all) took {timer.format(timer.elapsed_since_last())}")


# Give information about build time
print(f"Build image2image in {timer()}")
