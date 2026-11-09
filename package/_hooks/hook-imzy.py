from pathlib import Path

from koyo.system import IS_MAC, IS_WIN
from PyInstaller.utils.hooks import (
    can_import_module,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


def filter_imzy_data(data: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep imzy resources and shared libraries supported by the build platform."""
    filtered = []
    for src, module in data:
        path = Path(src)
        destination = module.replace("\\", "/")
        is_shared_object = path.suffix == ".so" or ".so." in path.name
        is_library = path.suffix in {".dll", ".dylib"} or is_shared_object
        if not is_library:
            filtered.append((src, module))
            continue
        if destination.startswith("imzy/_readers/waters") and not IS_WIN:
            continue
        if destination.startswith("imzy/_readers/bruker") and IS_MAC:
            continue
        if path.suffix == ".dll" and not IS_WIN:
            continue
        if path.suffix == ".dylib" and not IS_MAC:
            continue
        if is_shared_object and IS_WIN:
            continue
        filtered.append((src, module))
    return filtered


if not can_import_module("imzy._imzml_rust"):
    raise RuntimeError("The imzy Rust extension is required. Run the packaging installer with --update-deps.")

hiddenimports = ["imzy", "imzy._imzml_rust", *collect_submodules("pytsf")]
datas = (
    filter_imzy_data(collect_data_files("imzy", excludes=["**/*.dll", "**/*.dylib", "**/*.so", "**/*.so.*"]))
    + copy_metadata("imzy")
    + copy_metadata("pytsf")
)
binaries = filter_imzy_data(
    collect_dynamic_libs("imzy", search_patterns=["*.dll", "*.dylib", "lib*.so", "*.so.*"])
) + collect_dynamic_libs("pytsf")
