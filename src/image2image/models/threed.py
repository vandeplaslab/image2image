"""Three-D model."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from natsort import natsorted
from pydantic import Field

from image2image.config import CONFIG
from image2image.models.base import BaseModel
from image2image.models.utilities import _get_paths, _read_config_from_file
from image2image.utils.transform import combined_transform

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader


SCHEMA_VERSION: str = "1.0"


def as_icon(value: bool) -> str:
    """Convert to icon."""
    return "true" if value else "false"


class RegistrationImage(BaseModel):
    """Single image."""

    path: Path = Field(title="Path")
    key: str = Field("", title="Key")
    selected: bool = Field(False, title="Selected")
    keep: bool = Field(True, title="Keep")
    image_order: int = Field(0, title="Image order")
    group_id: int = Field(-1, title="Group ID")
    rotate: float = Field(0, title="Rotate")
    translate_x: float = Field(0, title="Translate X")
    translate_y: float = Field(0, title="Translate Y")
    flip_lr: bool = Field(False, title="Flip left-right")
    is_reference: bool = Field(False, title="Is reference")
    scale: tuple[float, float] = Field((1.0, 1.0), title="Scale")
    lock: bool = Field(False, title="Lock")

    @classmethod
    def from_reader(cls, reader: "BaseReader") -> "RegistrationImage":
        """Initialize from reader."""
        return cls(path=reader.path, key=reader.key, scale=reader.scale_for_pyramid(-1))  # type: ignore[call-arg]

    def update_from_reader(self, reader: "BaseReader") -> None:
        """Update from reader."""
        self.path = reader.path
        self.key = reader.key
        self.scale = reader.scale_for_pyramid(-1)

    def apply_rotate(self, which: str) -> None:
        """Apply rotation."""
        if which == "left":
            self.rotate -= CONFIG.rotate_step_size
        else:
            self.rotate += CONFIG.rotate_step_size

    def apply_translate(self, which: str) -> None:
        """Apply rotation."""
        if which == "up":
            self.translate_y -= CONFIG.translate_step_size
        elif which == "down":
            self.translate_y += CONFIG.translate_step_size
        elif which == "left":
            self.translate_x -= CONFIG.translate_step_size
        else:
            self.translate_x += CONFIG.translate_step_size

    def affine(self, shape: tuple[int, int]) -> np.ndarray:
        """Calculate affine transformation."""
        return combined_transform(
            shape,
            self.scale,
            self.rotate,
            (self.translate_y, self.translate_x),
            self.flip_lr,
        )

    def to_table(self) -> list:  # bool, int, str, int, int, float, float, float, bool]:
        """Convert to table."""
        return [
            self.selected,
            self.key,
            self.keep,
            self.lock,
            as_icon(self.is_reference),
            self.group_id,
            self.image_order,
            self.rotate,
            self.translate_x,
            self.translate_y,
            self.flip_lr,
        ]

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "path": str(self.path),
            "key": self.key,
            "selected": self.selected,
            "keep": self.keep,
            "is_reference": self.is_reference,
            "group_id": self.group_id,
            "image_order": self.image_order,
            "rotate": self.rotate,
            "translate_x": self.translate_x,
            "translate_y": self.translate_y,
            "flip_lr": self.flip_lr,
            "scale": self.scale,
        }


class Registration(BaseModel):
    """Registration metadata."""

    images: ty.Dict[str, RegistrationImage] = Field(default_factory=dict, title="Images")
    reference: ty.Optional[str] = Field(None, title="Reference")

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "version": SCHEMA_VERSION,
            "tool": "threed",
            "images": {key: image.to_dict() for key, image in self.images.items()},
            "reference": self.reference,
        }

    def model_iter(self) -> ty.Generator[RegistrationImage, None, None]:
        """Iterate over models."""
        yield from self.images.values()

    def group_to_key_map(self) -> dict[int, list[str]]:
        """Get mapping of group id to dataset."""
        group_to_key: dict[int, list[str]] = {}
        for key, image in self.images.items():
            group_to_key.setdefault(image.group_id, []).append(key)
        # sort using natural sort
        for group_id in natsorted(group_to_key):
            keys = group_to_key[group_id]
            group_to_key[group_id] = natsorted(keys)
        return group_to_key

    def is_grouped(self) -> bool:
        """Check whether images are grouped."""
        if not self.images:
            return False
        return any(model.group_id != -1 for model in self.images.values())

    def is_ordered(self) -> bool:
        """Check whether images are ordered."""
        if not self.images:
            return False
        return any(model.image_order != 0 for model in self.images.values())

    def reorder(self) -> None:
        """Apply ordering."""
        group_to_key_map = self.group_to_key_map()
        if group_to_key_map:
            # apply ordering
            index = 0
            for keys in group_to_key_map.values():
                for key in keys:
                    self.images[key].image_order = index
                    index += 1

    def key_iter(self) -> ty.Generator[str, None, None]:
        """Iterate over keys according to the `image_order` attribute."""
        order = sorted(self.images, key=lambda key: self.images[key].image_order)
        yield from order


def load_from_file(path: PathLike, validate_paths: bool = True) -> tuple[list[Path], list[Path], dict]:
    """Load from file."""
    path = Path(path)
    data = _read_config_from_file(path)
    if data["tool"] != "threed":
        raise ValueError(f"Invalid tool: {data['tool']}")
    # load images
    paths, missing_paths = [], []
    if validate_paths:
        paths = [Path(x["path"]) for x in data["images"].values()]
        paths, missing_paths = _get_paths(paths)
    return paths, missing_paths, data


def remove_if_not_present(config: dict, keys: list[str]) -> dict:
    """Remove if not present."""
    to_remove = []
    for item in config["images"].values():
        if item["key"] not in keys:
            to_remove.append(item["key"])
    for key in to_remove:
        del config["images"][key]
    return config
