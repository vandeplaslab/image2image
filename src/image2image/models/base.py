"""Base model."""

import typing as ty
from pathlib import Path

from koyo.typing import PathLike
from pydantic import BaseModel as _BaseModel


class BaseModel(_BaseModel):
    """Base model."""

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    def update(self, **kwargs: ty.Dict) -> None:
        """Update transformation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        raise NotImplementedError("Must implement method")

    def to_json(self, path: PathLike) -> None:
        """Export data as JSON."""
        from koyo.json import write_json_data

        path = Path(path)
        write_json_data(path, self.to_dict())

    @classmethod
    def from_json(cls, path: PathLike) -> "BaseModel":
        """Create from JSON."""
        from koyo.json import read_json_data

        path = Path(path)
        return cls.from_dict(read_json_data(path))

    def to_toml(self, path: PathLike) -> None:
        """Export data as TOML."""
        from koyo.toml import write_toml_data

        path = Path(path)
        write_toml_data(path, self.to_dict())

    @classmethod
    def from_toml(cls, path: PathLike) -> "BaseModel":
        """Create from TOML."""
        from koyo.toml import read_toml_data

        path = Path(path)
        return cls.from_dict(read_toml_data(path))

    @classmethod
    def from_dict(cls, data: dict) -> "BaseModel":
        """Create from dict."""
        raise NotImplementedError("Must implement method")

    @classmethod
    def from_file(cls, path: PathLike) -> "BaseModel":
        """Create from file."""
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".toml":
            return cls.from_toml(path)
        raise ValueError(f"Unknown file format: '{path.suffix}'")
