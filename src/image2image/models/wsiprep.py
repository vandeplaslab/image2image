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

    def affine(self, shape: tuple[int, int], scale: ty.Optional[tuple[float, float]] = None) -> np.ndarray:
        """Calculate affine transformation."""
        if scale is None:
            scale = self.scale
        return combined_transform(
            shape,
            scale,
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


class RegistrationGroup(BaseModel):
    """Registration group."""

    keys: list[str] = Field(default_factory=list, title="Keys")
    group_id: int = Field(-1, title="Group ID")
    mask_bbox: ty.Optional[tuple[int, int, int, int]] = Field(None, title="Mask bounding box")

    def clear(self) -> None:
        """Remove existing keys."""
        self.keys = []

    def to_iwsireg(self, output_dir: str, name: str) -> None:
        """Generate iwsireg configuration file."""

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "keys": self.keys,
            "group_id": self.group_id,
            "mask_bbox": self.mask_bbox,
        }


class Registration(BaseModel):
    """Registration metadata."""

    images: ty.Dict[str, RegistrationImage] = Field(default_factory=dict, title="Images")
    groups: ty.Dict[int, RegistrationGroup] = Field(default_factory=dict, title="Groups")

    def regroup(self) -> None:
        """Create groups based on the group_id attribute.

        Overwrite existing groups if those exist, although we don't remove the mask_bbox information.
        """
        existing_groups = self.groups
        self.groups = {}
        for image in self.images.values():
            if image.group_id == -1:
                continue
            if image.group_id in existing_groups:
                group = existing_groups[image.group_id]
                group.clear()
            elif image.group_id not in self.groups:
                group = RegistrationGroup(group_id=image.group_id)  # type: ignore[call-arg]
            else:
                group = self.groups[image.group_id]
            group.keys.append(image.key)
            self.groups[image.group_id] = group

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "version": SCHEMA_VERSION,
            "tool": "wsiprep",
            "images": {key: image.to_dict() for key, image in self.images.items()},
            "groups": {key: group.to_dict() for key, group in self.groups.items()},
        }

    def is_valid(self) -> bool:
        """Check whether there is any data in the model."""
        return len(self.images) > 0

    def append(self, reader: "BaseReader") -> None:
        """Append reader."""
        self.images[reader.key] = RegistrationImage.from_reader(reader)

    def remove(self, key: str) -> None:
        """Remove key."""
        del self.images[key]

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

    def get_reference_for_group(self, group_id: ty.Optional[int]) -> ty.Optional[str]:
        """Return name of the reference image for a group."""
        if group_id is None:
            for key, image in self.images.items():
                if image.is_reference:
                    return key
        else:
            for key, image in self.images.items():
                if image.group_id == group_id and image.is_reference:
                    return key
            return None

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
    if data["tool"] != "wsiprep":
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
        # remove it from images
        image = config["images"].pop(key)
        # also, potentially remove it from group
        if "groups" in image:
            if image["group_id"] != -1 and image["group_id"] in image["groups"]:
                if key in image["groups"][image["group_id"]]["keys"]:
                    index = image["groups"][image["group_id"]]["keys"].index(key)
                    image["groups"][image["group_id"]]["keys"].pop(index)
    return config
