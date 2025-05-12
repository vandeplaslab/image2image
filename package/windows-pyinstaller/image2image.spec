"""PyInstaller setup script."""

import os
from pathlib import Path

import debugpy._vendored
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
from image2image.assets import ICON_ICO as ICON_APP_ICO

# TODO change to it's own icon
from image2image.assets import ICON_ICO as ICON_REG_ICO

block_cipher = None
# allowed values: all, imports, bootloader, noarchive
DEBUG_MODE = os.getenv("PYINSTALLER_DEBUG", "imports")
print("DEBUG_MODE", DEBUG_MODE)
# allowed values: debug, info, warning, error, critical
LOG_MODE = os.getenv("PYINSTALLER_LOG", "DEBUG")
print("LOG_MODE", LOG_MODE)
# allowed values: hide-early, minimize-late, minimize-early, hide-late
CONSOLE_MODE = os.getenv("PYINSTALLER_CONSOLE", "hide-early")
print("CONSOLE_MODE", CONSOLE_MODE)


# allowed values: true, false
BUILD_I2REG = os.getenv("IMAGE2IMAGE_BUILD_I2REG", "true")
print("BUILD_I2REG", BUILD_I2REG)
BUILD_REGISTER = os.getenv("IMAGE2IMAGE_BUILD_REGISTER", "true")
print("BUILD_REGISTER", BUILD_REGISTER)
BUILD_VIEWER = os.getenv("IMAGE2IMAGE_BUILD_VIEWER", "true")
print("BUILD_VIEWER", BUILD_VIEWER)
BUILD_ELASTIX = os.getenv("IMAGE2IMAGE_BUILD_ELASTIX", "true")
print("BUILD_ELASTIX", BUILD_ELASTIX)

FILE_DIR = Path.cwd()
BASE_DIR = FILE_DIR.parent.parent
GITHUB_DIR = BASE_DIR.parent


def _make_analysis(path: str, ui: bool = True):
    print(f"Make analysis for {path}")
    datas = []
    hiddenimports = []
    if ui:
        datas = (
            []
            + collect_data_files("qtextra")
            + collect_data_files("qtextraplot")
            + collect_data_files("image2image")
            + collect_data_files("napari")
            + collect_data_files("freetype")
            + collect_data_files("glasbey")
            + collect_data_files("sklearn")
            + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")]
        )
        hiddenimports = [
            "freetype",
            "six",
            "pkg_resources",
            "glasbey",
        ]

    return Analysis(
        [path],
        binaries=[],
        datas=datas
        + collect_data_files("ome_types")
        + collect_data_files("distributed")
        + collect_data_files("xmlschema"),
        hiddenimports=hiddenimports,
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


def _make_app(path: str, name: str, icon: Path = ICON_APP_ICO, ui: bool = True):
    analysis = _make_analysis(str(path), ui=ui)
    print(f"Analysis ({name}) took {timer.format(timer.elapsed_since_last())}")
    pyz = PYZ(analysis.pure)
    print(f"PYZ ({name}) took {timer.format(timer.elapsed_since_last())}")
    exe = _make_exe(pyz, analysis, name, icon=icon)
    print(f"EXE ({name}) took {timer.format(timer.elapsed_since_last())}")
    return exe, analysis


# main app / launcher
with MeasureTimer() as timer:
    extra_args = ()

    # app
    launcher_exe, launcher_analysis = _make_app(
        BASE_DIR / "src" / "image2image" / "__main__.py",
        "image2image",
        icon=ICON_APP_ICO,
    )

    # viewer
    if BUILD_VIEWER == "true":
        viewer_exe, viewer_analysis = _make_app(
            BASE_DIR / "src" / "image2image" / "__main_viewer__.py",
            "i2viewer",
            icon=ICON_APP_ICO,
        )
        extra_args += (
            viewer_exe,
            viewer_analysis.binaries,
            viewer_analysis.zipfiles,
            viewer_analysis.datas,
        )
    # register
    if BUILD_REGISTER == "true":
        register_exe, register_analysis = _make_app(
            BASE_DIR / "src" / "image2image" / "__main_register__.py",
            "i2register",
            icon=ICON_APP_ICO,
        )
        extra_args += (
            register_exe,
            register_analysis.binaries,
            register_analysis.zipfiles,
            register_analysis.datas,
        )
    # elastix
    if BUILD_ELASTIX == "true":
        elastix_exe, elastix_analysis = _make_app(
            BASE_DIR / "src" / "image2image" / "__main_elastix__.py",
            "i2elastix",
            icon=ICON_APP_ICO,
        )
        extra_args += (
            elastix_exe,
            elastix_analysis.binaries,
            elastix_analysis.zipfiles,
            elastix_analysis.datas,
        )

    # image2image-reg
    if BUILD_I2REG == "true":
        reg_exe, reg_analysis = _make_app(
            GITHUB_DIR / "image2image-reg" / "src" / "image2image_reg" / "__main__.py",
            "i2reg",
            icon=ICON_APP_ICO,
            ui=False,
        )
        extra_args += (
            reg_exe,
            reg_analysis.binaries,
            reg_analysis.zipfiles,
            reg_analysis.datas,
        )

    # collect all
    image2image_coll = COLLECT(
        # launcher
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.zipfiles,
        launcher_analysis.datas,
        # other apps
        *extra_args,
        # other options
        strip=False,
        debug=DEBUG_MODE,
        upx=True,
        name="image2image",
    )
    print(f"COLLECT (all) took {timer.format(timer.elapsed_since_last())}")


# Give information about build time
print(f"Build image2image in {timer()}")
