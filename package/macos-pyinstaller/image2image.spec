"""PyInstaller setup script."""
import os
from pathlib import Path
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
from koyo.timer import MeasureTimer

block_cipher = None

def _make_analysis(path: str):
    return Analysis(
        [path],
        binaries=[],
        datas=[]
        + collect_data_files("qtextra")
        + collect_data_files("napari")
        + collect_data_files("xmlschema")
        + collect_data_files("ome_types")
        + collect_data_files("distributed")
        + collect_data_files("imzy")
        + collect_data_files("napari")
        + collect_data_files("image2image")
        + collect_data_files("freetype")
        + collect_data_files("xmlschema")

        + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=[]
        + [
            "pkg_resources",
            "six",
            "psygnal",
            "psygnal._signal",
            "qtpy",
            "freetype",
            "magicgui.backends._qtpy",
            "imzy",
        ],
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
        console=False,
        windowed=True,
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
        entitlements_file="entitlements.plist",
        bundle_identifier="com.vandeplaslab.image2image",
    )


# main app / launcher
with MeasureTimer() as timer:
    launcher_analysis = _make_analysis("../../src/image2image/__main__.py")
    print(f"Analysis took {timer.format(timer.elapsed_since_last())}")

    launcher_pyz = PYZ(launcher_analysis.pure)
    print(f"PYZ took {timer.format(timer.elapsed_since_last())}")

    launcher_exe = _make_exe(launcher_pyz, launcher_analysis, "image2image_")
    print(f"EXE took {timer.format(timer.elapsed_since_last())}")

    # collect all
    image2image_coll = COLLECT(
        # launcher
        launcher_exe,
        launcher_analysis.binaries,
        launcher_analysis.zipfiles,
        launcher_analysis.datas,
        strip=False,
        debug="all",
        upx=True,
        name="image2image",
    )
    print(f"COLLECT took {timer.format(timer.elapsed_since_last())}")

    image2imag_app = BUNDLE(
        image2image_coll,
        # image2image_a.binaries,
        # image2image_a.zipfiles,
        # image2image_a.datas,
        name="image2image.app",
        icon=ICON_ICO,
        bundle_identifier="com.vandeplaslab.image2image",
        info_plist={"NSHighResolutionCapable": "True"},
    )
    print(f"BUNDLE took {timer.format(timer.elapsed_since_last())}")

# Give information about build time
print(f"Build image2image in {timer()}")
