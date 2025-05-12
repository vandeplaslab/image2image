"""Universal image2image pyinstaller spec file."""
from __future__ import annotations
import os
from pathlib import Path
import inspect

import debugpy._vendored
import imagecodecs
import napari
import qtpy
from koyo.timer import MeasureTimer
from koyo.system import IS_MAC
from PyInstaller.building.build_main import COLLECT, EXE, MERGE, PYZ, TOC, Analysis
from PyInstaller.utils.hooks import collect_data_files

import image2image
from image2image.assets import ICON_ICO as ICON_APP_ICO

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


# Get the parent directory of this file
parent = Path(inspect.getfile(lambda: None)).parent.resolve()
hooks_dir = parent / "_hooks"
assert hooks_dir.exists(), "Hooks directory does not exist"
runtimehooks_dir = parent / "_runtimehooks"

assert runtimehooks_dir.exists(), "Runtime hooks directory does not exist"
runtimehooks = [str(f) for f in runtimehooks_dir.glob("hook-*.py")]
print(runtimehooks)

script_file = parent.parent / "src" / "image2image" / "__main__.py"
assert script_file.exists(), "Script file does not exist"


def _make_analysis(path: str | Path):
    return Analysis(
        [str(path)],
        binaries=[],
        datas=[]
            + collect_data_files("qtextra")
            + collect_data_files("qtextraplot")
            + collect_data_files("image2image")
            + collect_data_files("napari")
            + collect_data_files("xmlschema")
            + collect_data_files("ome_types")
            + collect_data_files("distributed")
            + collect_data_files("freetype")
            + collect_data_files("glasbey")
            + collect_data_files("sklearn")
            + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=[
            "freetype",
            "six",
            "pkg_resources",
            "glasbey",
        ],
        hookspath=[str(hooks_dir)],
        runtime_hooks=runtimehooks,
        excludes=[
            "tcl",
            "Tkconstants",
            "Tkinter",
        ],
        cipher=block_cipher,
    )


def _make_exe(pyz: PYZ, analysis: Analysis, name: str, icon: Path = ICON_APP_ICO):
    """Make the executable."""
    kws = {}
    if IS_MAC:
        kws = {
            "entitlements_file": str(parent / "macos-pyinstaller" / "entitlements.plist"),
            "bundle_identifier": "com.vandeplaslab.image2image",
        }

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
        **kws,
    )


def _print_timing(name: str, timer_: MeasureTimer, since_last: bool=True) -> None:
    try:
        print(f"{name} took {timer_(since_last=since_last)}")
    except UnicodeEncodeError:
        print(f"{name} took {timer_(since_last=since_last).encode('utf-8')}")


# main app / launcher
with MeasureTimer() as timer:
    launcher_analysis = _make_analysis(script_file)
    _print_timing("Analysis", timer)

    launcher_pyz = PYZ(launcher_analysis.pure)
    _print_timing("PYZ", timer)

    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image_")
    _print_timing("EXE", timer)

    image2image_coll = COLLECT(
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.zipfiles,
        launcher_analysis.datas,
        strip=False,
        debug=DEBUG_MODE,
        upx=True,
        name="image2image",
    )
    _print_timing("COLLECT", timer)

    if IS_MAC:
        image2imag_app = BUNDLE(
            image2image_coll,
            name="image2image.app",
            icon=ICON_APP_ICO,
            bundle_identifier="com.vandeplaslab.image2image",
            info_plist={
                "CFBundleIdentifier": "com.vandeplaslab.image2image",
                "CFBundleName": "autoims",
                "NSPrincipalClass": "NSApplication",
                "NSRequiresAquaSystemAppearance": "Yes",
                "NSHighResolutionCapable": "True",
                "LSHandlerRank": "Default",
                "NSHumanReadableCopyright": "Copyright © 2023-2025 Van de Plas lab. All Rights Reserved",
                "LSMinimumSystemVersion": "10.13",
                "CFBundleShortVersionString": "0.0.1",
            },
        )
        _print_timing("BUNDLE", timer)

# Give information about build time
_print_timing("APP", timer, since_last=False)
