"""PyInstaller setup script."""

import os
from pathlib import Path

import debugpy._vendored
import imagecodecs
import napari
import qtpy
from koyo.timer import MeasureTimer
from koyo.pyinstaller import load_hooks, get_runtime_hooks
from PyInstaller.building.build_main import BUNDLE, COLLECT, EXE, PYZ, Analysis

import image2image
from image2image.assets import ICON_ICO

# allowed values: all, imports, bootloader, noarchive
debug_mode = os.getenv("PYINSTALLER_DEBUG", "0").strip().lower()
DEBUG_MODE = False if debug_mode in {"", "0", "false", "off", "no", "none"} else "all"
print("DEBUG_MODE", DEBUG_MODE)
# allowed values: debug, info, warning, error, critical
LOG_MODE = os.getenv("PYINSTALLER_LOG", "DEBUG")
print("LOG_MODE", LOG_MODE)
# allowed values: hide-early, minimize-late, minimize-early, hide-late
CONSOLE_MODE = os.getenv("PYINSTALLER_CONSOLE", "hide-early")
print("CONSOLE_MODE", CONSOLE_MODE)

FILE_DIR = Path(SPECPATH)
BASE_DIR = FILE_DIR.parent.parent
GITHUB_DIR = BASE_DIR.parent

HOOKS_DIR = BASE_DIR / "package" / "_hooks"
assert HOOKS_DIR.exists(), f"Hooks directory does not exist - {HOOKS_DIR}"
RUNTIMEHOOKS_DIR = BASE_DIR / "package" / "_runtimehooks"
assert RUNTIMEHOOKS_DIR.exists(), f"Runtime hooks directory does not exist - {RUNTIMEHOOKS_DIR}"

PY_SCRIPT_FILE = BASE_DIR / "src" / "image2image" / "__main__.py"
assert PY_SCRIPT_FILE.exists(), f"Script file does not exist - {PY_SCRIPT_FILE}"


def _make_analysis(path: str):
    print(f"Make analysis for {path}")

    # manually load hooks
    hiddenimports, datas, binaries = load_hooks(HOOKS_DIR)
    print(f"  hiddenimports: {len(hiddenimports)}")
    print(f"  datas: {len(datas)}")
    print(f"  binaries: {len(binaries)}")
    runtime_hooks = get_runtime_hooks(RUNTIMEHOOKS_DIR)
    print(f"  runtime_hooks: {runtime_hooks}")

    return Analysis(
        [str(path)],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        runtime_hooks=runtime_hooks,
        excludes=["tcl", "Tkconstants", "Tkinter", "pyspa"],
    )


def _make_exe(pyz: PYZ, analysis: Analysis, name: str):
    """Make the executable."""
    return EXE(
        pyz,
        analysis.scripts,
        exclude_binaries=True,
        name=name,
        debug=DEBUG_MODE,
        strip=False,
        upx=True,
        console=False,
        windowed=True,
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
        entitlements_file=str(FILE_DIR / "entitlements.plist"),
        bundle_identifier="com.vandeplaslab.image2image",
    )


# main app / launcher
with MeasureTimer() as timer:
    launcher_analysis = _make_analysis(PY_SCRIPT_FILE)
    print(f"Analysis took {timer.format(timer.elapsed_since_last())}")

    launcher_pyz = PYZ(launcher_analysis.pure)
    print(f"PYZ took {timer.format(timer.elapsed_since_last())}")

    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image_")
    print(f"EXE took {timer.format(timer.elapsed_since_last())}")

    image2image_coll = COLLECT(
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.datas,
        strip=False,
        debug=DEBUG_MODE,
        upx=True,
        name="image2image",
    )
    print(f"COLLECT took {timer.format(timer.elapsed_since_last())}")

    image2imag_app = BUNDLE(
        image2image_coll,
        name="image2image.app",
        icon=ICON_ICO,
        bundle_identifier="com.vandeplaslab.image2image",
        info_plist={
            "CFBundleIdentifier": "com.vandeplaslab.image2image",
            "CFBundleName": "image2image",
            "NSPrincipalClass": "NSApplication",
            "NSRequiresAquaSystemAppearance": "Yes",
            "NSHighResolutionCapable": "True",
            "LSHandlerRank": "Default",
            "NSHumanReadableCopyright": "Copyright 2023-2026 Van de Plas lab. All Rights Reserved",
            "LSMinimumSystemVersion": "10.13",
            "CFBundleShortVersionString": image2image.__version__,
            "CFBundleVersion": image2image.__version__,
        },
    )
    print(f"BUNDLE took {timer.format(timer.elapsed_since_last())}")

# Give information about build time
print(f"Build image2image in {timer()}")
