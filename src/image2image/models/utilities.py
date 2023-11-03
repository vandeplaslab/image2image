"""Utilities."""
import typing as ty
from pathlib import Path

from koyo.typing import PathLike


def _read_config_from_file(path: PathLike) -> ty.Dict[str, ty.Any]:
    """Read config data from file."""
    path = Path(path)
    if path.suffix not in [".json", ".toml"]:
        raise ValueError(f"Unknown file format: {path.suffix}")

    if path.suffix == ".json":
        from koyo.json import read_json_data

        data: ty.Dict = read_json_data(path)
    else:
        from koyo.toml import read_toml_data

        data: ty.Dict = read_toml_data(path)
    return data


def _remove_missing_from_dict(affine: ty.Dict[str, ty.Any], paths: ty.List[PathLike]) -> ty.Dict[str, ty.Any]:
    """Remove elements that are not present in the list of paths from the dictionary."""
    names = [Path(path).name for path in paths]
    for name in list(affine.keys()):
        if name not in names:
            del affine[name]
    return affine


def _get_paths(paths: ty.List[PathLike]) -> ty.Tuple[ty.Optional[ty.List[Path]], ty.Optional[ty.List[Path]]]:
    _paths_exist, _paths_missing = [], []
    for path in paths:
        path = Path(path)
        try:
            if path.exists():
                _paths_exist.append(path)
            else:
                _paths_missing.append(path)
        except PermissionError:
            _paths_missing.append(path)
    if not _paths_exist:
        _paths_exist = []
    return _paths_exist, _paths_missing
