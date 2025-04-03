"""PyInstaller setup script."""

import os
import time
from pathlib import Path

import imagecodecs
import napari
import qtpy
from koyo.timer import MeasureTimer
from PyInstaller.building.build_main import COLLECT, EXE, MERGE, PYZ, TOC, Analysis
from PyInstaller.utils.hooks import (
    PY_IGNORE_EXTENSIONS,
    collect_data_files,
    get_package_paths,
    remove_prefix,
)

import image2image
from image2image.assets import ICON_ICO

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
    return Analysis(
        [path],
        binaries=[],
        datas=[],
        hiddenimports=[],
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
        console=True,  # False,
        hide_console="hide-early",
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
    )


# main app / launcher
with MeasureTimer() as timer:
    # app
    launcher_analysis = _make_analysis("../../src/image2image/__main__.py")
    print(f"Analysis (app) took {timer.format(timer.elapsed_since_last())}")
    launcher_pyz = PYZ(launcher_analysis.pure)
    print(f"PYZ (app) took {timer.format(timer.elapsed_since_last())}")
    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image")
    print(f"EXE (app) took {timer.format(timer.elapsed_since_last())}")

    # registration
    reg_analysis = _make_analysis("../../src/image2image_reg/__main__.py")
    print(f"Analysis (reg) took {timer.format(timer.elapsed_since_last())}")
    reg_pyz = PYZ(reg_analysis.pure)
    print(f"PYZ (reg) took {timer.format(timer.elapsed_since_last())}")
    reg_exe = _make_exe(reg_pyz, reg_analysis, "i2reg")
    print(f"EXE (reg) took {timer.format(timer.elapsed_since_last())}")

    # collect all
    image2image_coll = COLLECT(
        # launcher
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.zipfiles,
        launcher_analysis.datas,
        # reg
        reg_exe,
        reg_analysis.binaries,
        reg_analysis.zipfiles,
        reg_analysis.datas,
        # other options
        strip=False,
        debug="all",
        upx=True,
        name="image2image",
    )
    print(f"COLLECT (all) took {timer.format(timer.elapsed_since_last())}")


# Give information about build time
time_end = time.time()
print(f"Build image2image in {time_end - time_start:.2f} seconds\n")
