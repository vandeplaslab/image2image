"""PyInstaller setup script."""

import os
from pathlib import Path
import inspect

import debugpy._vendored
import imagecodecs
import napari
import qtpy
from koyo.timer import MeasureTimer
from PyInstaller.building.build_main import COLLECT, EXE, MERGE, PYZ, TOC, Analysis
from PyInstaller.utils.hooks import collect_data_files

import image2image
from image2image.assets import ICON_ICO

block_cipher = None
DEBUG_MODE = os.getenv("PYINSTALLER_DEBUG", "all")

# Get the parent directory of this file
parent = Path(inspect.getfile(lambda: None)).parent.resolve()
hooks_dir = parent.parent / "_hooks"
assert hooks_dir.exists(), "Hooks directory does not exist"
runtimehooks_dir = parent.parent / "_runtimehooks"

assert runtimehooks_dir.exists(), "Runtime hooks directory does not exist"
runtimehooks = [str(f) for f in runtimehooks_dir.glob("hook-*.py")]
print(runtimehooks)

script_file = parent.parent.parent / "src" / "image2image" / "__main__.py"
assert script_file.exists(), "Script file does not exist"


def _make_analysis(path: str):
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
        + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=["freetype", "six", "pkg_resources"],
        hookspath=[str(hooks_dir)],
        runtime_hooks=runtimehooks,
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
        debug=DEBUG_MODE,
        strip=False,
        upx=True,
        console=False,
        windowed=True,
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
        entitlements_file="entitlements.plist",
        bundle_identifier="com.vandeplaslab.image2image",
    )


# main app / launcher
with MeasureTimer() as timer:
    launcher_analysis = _make_analysis(script_file)
    print(f"Analysis took {timer.format(timer.elapsed_since_last())}")

    launcher_pyz = PYZ(launcher_analysis.pure)
    print(f"PYZ took {timer.format(timer.elapsed_since_last())}")

    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image_")
    print(f"EXE took {timer.format(timer.elapsed_since_last())}")

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
    print(f"COLLECT took {timer.format(timer.elapsed_since_last())}")

    image2imag_app = BUNDLE(
        image2image_coll,
        name="image2image.app",
        icon=ICON_ICO,
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
    print(f"BUNDLE took {timer.format(timer.elapsed_since_last())}")

# Give information about build time
print(f"Build image2image in {timer()}")
