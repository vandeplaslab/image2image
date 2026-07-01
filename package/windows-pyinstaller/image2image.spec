"""PyInstaller setup script."""

# We just need to import them
import debugpy._vendored
import imagecodecs
import napari
import qtpy

# Actually needed
import os
from pathlib import Path
from koyo.timer import MeasureTimer
from koyo.pyinstaller import load_hooks, get_runtime_hooks
from PyInstaller.building.build_main import COLLECT, EXE, PYZ, Analysis

# App imports
import image2image
from image2image.assets import ICON_ICO as ICON_APP_ICO

# allowed values: all, imports, bootloader, noarchive
DEBUG_MODE = os.getenv("PYINSTALLER_DEBUG", "imports")
print("DEBUG_MODE", DEBUG_MODE)
# allowed values: debug, info, warning, error, critical
LOG_MODE = os.getenv("PYINSTALLER_LOG", "DEBUG")
print("LOG_MODE", LOG_MODE)
# allowed values: hide-early, minimize-late, minimize-early, hide-late
CONSOLE_MODE = os.getenv("PYINSTALLER_CONSOLE", "hide-early")
print("CONSOLE_MODE", CONSOLE_MODE)

FILE_DIR = Path.cwd()
BASE_DIR = FILE_DIR.parent.parent
GITHUB_DIR = BASE_DIR.parent

HOOKS_DIR = BASE_DIR / "package" / "_hooks"
assert HOOKS_DIR.exists(), f"Hooks directory does not exist - {HOOKS_DIR}"
RUNTIMEHOOKS_DIR = BASE_DIR / "package" / "_runtimehooks"
assert RUNTIMEHOOKS_DIR.exists(), f"Runtime hooks directory does not exist - {RUNTIMEHOOKS_DIR}"

PY_SCRIPT_FILE = GITHUB_DIR / "image2image" / "src" / "image2image" / "__main__.py"
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
        [path],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        runtime_hooks=runtime_hooks,
        excludes=["tcl", "Tkconstants", "Tkinter", "black", "mypy"],
        cipher=None,
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


def _make_app(path: str, name: str, icon: Path = ICON_APP_ICO):
    analysis = _make_analysis(str(path))
    print(f"Analysis ({name}) took {timer.format(timer.elapsed_since_last())}")
    pyz = PYZ(analysis.pure)
    print(f"PYZ ({name}) took {timer.format(timer.elapsed_since_last())}")
    exe = _make_exe(pyz, analysis, name, icon=icon)
    print(f"EXE ({name}) took {timer.format(timer.elapsed_since_last())}")
    return exe, analysis


# main app / launcher
with MeasureTimer() as timer:

    # app
    launcher_exe, launcher_analysis = _make_app(PY_SCRIPT_FILE, "image2image", icon=ICON_APP_ICO)

    # collect all
    image2image_coll = COLLECT(
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
