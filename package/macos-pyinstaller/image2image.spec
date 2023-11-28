"""PyInstaller setup script."""
# -*- mode: python -*-
import os
from pathlib import Path
import time
from image2image.assets import ICON_ICO
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, TOC, BUNDLE, COLLECT
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
from koyo.timer import MeasureTimer
import qtpy
import napari
import image2image
import debugpy._vendored
import imagecodecs


block_cipher = None

# Extra imports that can sometimes be problematic
hiddenimports = (
    [
        # scipy
        "scipy",
        "scipy.sparse.csgraph._validation",
        "scipy.linalg.cython_blas",
        "scipy.integrate",
        "scipy.special",
        "scipy.special._ufuncs_cxx",
        "scipy.special._ufuncs",
        "scipy.stats",
        "scipy._lib.messagestream",
        # numba - these actually don't work and the Jitclass must be disabled manually in numba
        "numba",
        "numba.core",
        "numba.experimental",
        "numba.experimental.jitclass",
        "numba.experimental.jitclass.base",
        "numba.experimental.jitclass.boxing",
        "numba.experimental.jitclass._box",
        "numba.core.typing.cffi_utils",
        # sklearn
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._cython_blas",
        # bokeh - found it in the past to cause trouble
        # "bokeh",
        # internal - let's be on the safe side
    ]
    + [f"imagecodecs.{y}" for y in (x if x[0] == "_" else f"_{x}" for x in imagecodecs._extensions())]
    + ["imagecodecs._shared"]
)

with MeasureTimer() as timer:
    image2image_a = Analysis(
        ["../../src/image2image/__main__.py"],
        binaries=[],
        datas=[]
        + collect_data_files("numba")
        + collect_data_files("qtextra")
        + collect_data_files("napari")
        + collect_data_files("napari_plot")
        + collect_data_files("xmlschema")
        + collect_data_files("ome_types")
        + collect_data_files("distributed")
        + collect_data_files("imagecodecs")
        + collect_data_files("image2image")
        + collect_data_files("vispy")
        + collect_data_files("freetype")
        + [(os.path.dirname(debugpy._vendored.__file__), "debugpy/_vendored")],
        hiddenimports=[]
        + hiddenimports
        + [
            "pkg_resources",
            "six",
            "psygnal",
            "psygnal._signal",
            "pyside2",
            "qtpy",
            "vispy.app.backends._pyside2",
            "freetype",
            "magicgui.backends._qtpy",
        ],
        hookspath=[],
        runtime_hooks=[
            "runtimehooks/hook-bundle.py",
            "runtimehooks/hook-multiprocessing.py",
        ],
        excludes=[] + ["tcl", "Tkconstants", "Tkinter"],
        cipher=block_cipher,
    )
    print(f"Analysis took {timer.format(timer.elapsed_since_last())}")

    image2image_pyz = PYZ(image2image_a.pure)
    print(f"PYZ took {timer.format(timer.elapsed_since_last())}")

    image2image_exe = EXE(
        image2image_pyz,
        image2image_a.scripts,
        exclude_binaries=True,
        name="image2image_",
        debug="all",
        strip=False,
        upx=True,
        console=True,
        bootloader_ignore_signals=False,
        icon=ICON_ICO,
        codesign_identity="vandeplaslab",
        entitlements_file="entitlements.plist",
    )
    print(f"EXE took {timer.format(timer.elapsed_since_last())}")

    image2image_coll = COLLECT(
        image2image_exe,
        image2image_a.binaries,
        image2image_a.zipfiles,
        image2image_a.datas,
        strip=False,
        debug="all",
        upx=True,
        name="image2image_",  # needed to avoid name clash with image2image
        codesign_identity="vandeplaslab",
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
