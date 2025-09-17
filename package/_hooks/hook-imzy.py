from pathlib import Path
from koyo.system import IS_MAC, IS_LINUX, IS_WIN
from PyInstaller.utils.hooks import collect_data_files


def filter_imzy_data(data: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Filter out unwanted files from imspy_data package data."""
    filtered = []
    for src, module in data:
        src = Path(src)
        if module == "imzy/_readers/waters" and (IS_MAC or IS_LINUX):
            continue
        if module == "imzy/_readers/bruker":
            if IS_MAC:  # never available on MacOS
                continue
            elif IS_LINUX and src.suffix in {".dll", ".dylib"}:
                continue
            elif IS_WIN and src.suffix in {".so", ".dylib"}:
                continue
        filtered.append((src, module))
    return filtered


hiddenimports = ["imzy"]
datas = filter_imzy_data(collect_data_files("imzy"))
