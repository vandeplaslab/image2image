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
    from image2image_wsireg.workflows import IWsiReg


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
        if self.rotate > 360:
            self.rotate = 0
        elif self.rotate < 0:
            self.rotate = 360

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

    @classmethod
    def is_identity(cls, affine: np.ndarray) -> bool:
        """Check whether affine is identity."""
        return np.allclose(affine, np.eye(3))

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
            "lock": self.lock,
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

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "keys": self.keys,
            "group_id": self.group_id,
            "mask_bbox": self.mask_bbox,
        }

    def to_iwsireg(
        self,
        registration: "Registration",
        output_dir: PathLike,
        name: str,
        transformations: tuple[str, ...] = ("rigid", "affine"),
    ) -> Path:
        """Generate iwsireg configuration file."""
        from image2image_io._reader import get_simple_reader
        from image2image_wsireg.models import Preprocessing
        from image2image_wsireg.workflows import IWsiReg

        # retrieve images
        images = [registration.images[key] for key in self.keys]
        for image in images:
            if not image.path:
                continue
            assert image.path.exists(), f"Path does not exist: {image.path}"

        # sort images according to the image order, and if one is not specified, use the key
        if all(image.image_order == 0 for image in images):
            for index, image in enumerate(natsorted(images, key=lambda x: x.key)):
                image.image_order = index

        index_to_image, kind, reference = self.sort(registration)
        if len(index_to_image) == 1:
            raise ValueError("Cannot register a single image")

        # create iwsireg object
        path = Path(output_dir) / f"{name}.wsireg"
        obj = IWsiReg(name=name, output_dir=output_dir, cache=True, merge=True)
        for index, key in index_to_image.items():
            image = registration.images[key]
            reader = get_simple_reader(image.path)
            pre = Preprocessing.fluorescence()
            affine = image.affine(reader.image_shape, reader.scale)
            if not RegistrationImage.is_identity(affine):
                pre.affine = affine
            # add modality to the project
            obj.add_modality(
                f"c{index}",
                image.path,
                pixel_size=reader.resolution,
                preprocessing=pre,
                mask_bbox=self.mask_bbox if index == 0 else None,
            )
        # add registration paths
        if kind == "cascade":
            self._add_cascade_paths(obj, index_to_image, transformations=transformations)
        elif kind == "converge":
            if not reference:
                raise ValueError("Reference must be present to converge")
            self._add_converge_paths(obj, reference, index_to_image, transformations=transformations)

        # validate the registration
        if not obj.validate(allow_not_registered=True):
            raise ValueError("Invalid registration")
        # save the registration
        obj.save()
        return path

    def to_preview(self, registration: "Registration") -> str:
        """Generate preview of the registration paths."""
        # retrieve images
        images = [registration.images[key] for key in self.keys]

        # sort images according to the image order, and if one is not specified, use the key
        if all(image.image_order == 0 for image in images):
            for index, image in enumerate(natsorted(images, key=lambda x: x.key)):
                image.image_order = index

        index_to_image, kind, reference = self.sort(registration)
        if len(index_to_image) == 1:
            raise ValueError("Cannot register a single image")

        if kind == "cascade":
            return self._preview_cascade_paths(index_to_image, reference)
        elif kind == "converge":
            if not reference:
                raise ValueError("Reference must be present to converge")
            return self._preview_converge_paths(index_to_image, reference)
        return ""

    @staticmethod
    def _add_cascade_paths(
        obj: "IWsiReg",
        index_to_image: dict[int, str],
        transformations: tuple[str, ...] = ("rigid", "affine"),
    ) -> None:
        """Add cascade paths."""
        for image_index, _key in index_to_image.items():
            # no need to register the first image, we will register against it
            if image_index == 0:
                continue
            # register second image to the first - no need for through modality
            obj.add_registration_path(
                f"c{image_index}",
                "c0",
                transform=transformations,
                through=None if image_index == 1 else f"c{image_index - 1}",
            )

    @staticmethod
    def _preview_cascade_paths(index_to_image: dict[int, str], reference: ty.Optional[str]) -> str:
        """Generate preview of the registration paths."""
        lines = [f"<b>Reference: {reference} ({'c0' if reference else 'None'})</b><br>"]
        for image_index, key in index_to_image.items():
            # no need to register the first image, we will register against it
            if image_index == 0:
                continue
            # register second image to the first - no need for through modality
            if image_index == 1:
                lines.append(f"- {key} -> {reference}<br><b>(c{image_index} -> {'c0' if reference else 'None'})</b>")
            else:
                lines.append(
                    f"- {key} -> {reference} via {index_to_image[image_index - 1]}"
                    f"<br><b>(c{image_index} -> {'c0' if reference else 'None'} via c{image_index - 1})</b>"
                )
        return "<br>".join(lines)

    @staticmethod
    def _add_converge_paths(
        obj: "IWsiReg",
        reference: str,
        index_to_image: dict[int, str],
        transformations: tuple[str, ...] = ("rigid", "affine"),
    ) -> None:
        """Add cascade paths."""
        # find the index of reference in the index_to_image
        image_orders = list(index_to_image.keys())
        index = list(index_to_image.values()).index(reference)
        ref_index = image_orders[index]
        image_orders_before = image_orders[:index]
        image_orders_after = image_orders[index + 1 :]
        # first, we iterate over the images before the reference, where we create a registration path that goes like
        # this 0 -> ref via 1, 1 -> ref via 2, etc. until we reach the reference
        n = len(image_orders_before)
        for index, image_index in enumerate(image_orders_before):
            if index == ref_index:
                break
            obj.add_registration_path(
                f"c{image_index}",
                f"c{ref_index}",
                transform=transformations,
                through=None if index == n - 1 else f"c{image_orders_before[index + 1]}",
            )
        # then, we iterate over the images after the reference, where we create a registration path that goes like this
        # 3 -> ref via 2, 2 -> ref via 1, etc. until we reach the reference
        n = len(image_orders_after)
        for index, image_index in enumerate(reversed(image_orders_after)):
            if index == ref_index:
                break
            obj.add_registration_path(
                f"c{image_index}",
                f"c{ref_index}",
                transform=transformations,
                through=None if index == n - 1 else f"c{image_orders_after[index - 1]}",
            )

    @staticmethod
    def _preview_converge_paths(index_to_image: dict[int, str], reference: str) -> str:
        """Preview converge paths."""
        # find the index of reference in the index_to_image
        image_orders = list(index_to_image.keys())
        index = list(index_to_image.values()).index(reference)
        ref_index = image_orders[index]
        image_orders_before = image_orders[:index]
        image_orders_after = image_orders[index + 1 :]
        lines = [f"<b>Reference: {reference} (c{ref_index})</b><br>"]
        n = len(image_orders_before)
        for index, image_index in enumerate(image_orders_before):
            if index == ref_index:
                break
            if index == n - 1:
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference}<br><b>(c{image_index} -> c{ref_index})</b>"
                )
            else:
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference} via {index_to_image[image_index + 1]}"
                    f"<br><b>(c{image_index} -> c{ref_index} via c{index + 1})</b>"
                )
        n = len(image_orders_after)
        for index, image_index in enumerate(reversed(image_orders_after)):
            if index == ref_index:
                break
            if index == n - 1:
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference}<br><b>(c{image_index} -> c{ref_index})</b>"
                )
            else:
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference} via {index_to_image[image_index - 1]}"
                    f"<br><b>(c{image_index} -> c{ref_index} via c{index - 1})</b>"
                )
        return "<br>".join(lines)

    def sort(self, registration: "Registration", reset: bool = True) -> tuple[dict[int, str], str, ty.Optional[str]]:
        """Sort keys according to the following rules.

        - If no reference is present:
            - sort by the image_order if present
            - otherwise natural sorting by names
        - If reference is present, and it's the first image:
            - reference must be first
            - then sort by image_order if present
            - otherwise natural sorting by names
        - If reference is present but it's not the first image:
            e.g. if we had the following images 1, 2, ref, 4, 5, then we would have 1 -> 2 -> ref <- 4 <- 5
            - reference does not have to be first
            - all images must be pointed TO the reference
            - sort by image_order if present
            - otherwise natural sorting by names

        """
        # if no reference is present, sort by image_order if present, otherwise natural sorting by names
        if not self.keys:
            return {}, "", None

        kind = "cascade"
        reference = registration.get_reference_for_group(self.group_id)
        nat_keys = natsorted(self.keys)
        image_orders = [registration.images[key].image_order for key in nat_keys]
        if reset:
            image_orders = [0] * len(image_orders)

        # no reference is available, sort by image_order if present, otherwise natural sorting by names
        if reference is None:
            reference = nat_keys[0]

        if reference not in nat_keys:
            raise ValueError("There is a mismatch between the reference and the images")
        # if reference is present, and it's the first image, reference must be first,
        # then sort by image_order if present, otherwise natural sorting by names
        ref_index = nat_keys.index(reference)
        ref_order = image_orders[ref_index]
        if ref_index == 0:
            # image order has not been set OR reference is the first image
            if len(np.unique(image_orders)) == 1 or ref_order == np.min(image_orders):
                index_to_image = dict(enumerate(nat_keys))
            else:
                # order by the image order BUT make sure that reference is first in the stack
                nat_keys.pop(ref_index)
                image_orders = [registration.images[key].image_order for key in nat_keys]
                nat_keys = [reference, *nat_keys]
                image_orders = [np.min(image_orders) - 1, *image_orders]
                index_to_image = dict(zip(image_orders, nat_keys))
        # if reference is present but it's not the first image, reference does not have to be first, all images must
        # be pointed TO the reference, sort by image_order if present, otherwise natural sorting by names
        # e.g. if we had the following images 1, 2, ref, 4, 5, then we would have 1 -> 2 -> ref <- 4 <- 5
        # we need to sort the keys so that the reference is in the middle
        else:
            # first, we need to find the reference
            nat_keys_before = nat_keys[:ref_index]
            # image_orders[:ref_index]
            nat_keys_after = nat_keys[ref_index + 1 :]
            # image_orders[ref_index + 1 :]
            # ref_order -= 1
            index_to_image = {}
            index = 0
            for index, key in enumerate(reversed(nat_keys_before)):
                index_to_image[index] = key
            index_to_image[index + 1] = reference
            for index, key in enumerate(nat_keys_after, start=index + 2):
                index_to_image[index] = key
            # if len(np.unique(image_orders_before)) == 1:  # and len(image_orders_after) > 1:
            #     index_to_image_before = dict(enumerate(reversed(nat_keys_before)))
            # else:
            #     index_to_image_before = dict(zip(reversed(image_orders_before), reversed(nat_keys_before)))
            # if len(np.unique(image_orders_after)) == 1:  # and len(image_orders_after) > 1:
            #     index_to_image_after = dict(enumerate(nat_keys_after, start=))
            # else:
            #     index_to_image_after = dict(zip(image_orders_after, nat_keys_after))
            # index_to_image = {**index_to_image_before, ref_order: reference, **index_to_image_after}
            # print(index_to_image_before)
            # print(index_to_image_after)
            # print(ref_order)
            kind = "converge"
        return index_to_image, kind, reference


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
                group = existing_groups.pop(image.group_id)
                group.clear()
            elif image.group_id not in self.groups:
                group = RegistrationGroup(group_id=image.group_id)  # type: ignore[call-arg]
            else:
                group = self.groups[image.group_id]
            self.groups[image.group_id] = group
            group.keys.append(image.key)

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
        for group in self.groups.values():
            index_to_image, _, reference = group.sort(self)
            for index, key in index_to_image.items():
                self.images[key].image_order = index

        # group_to_key_map = self.group_to_key_map()
        # if group_to_key_map:
        #     # apply ordering
        #     index = 0
        #     for keys in group_to_key_map.values():
        #         for key in keys:
        #             self.images[key].image_order = index
        #             index += 1

    def key_iter(self) -> ty.Generator[str, None, None]:
        """Iterate over keys according to the `image_order` attribute."""
        order = sorted(self.images, key=lambda key: self.images[key].image_order)
        yield from order

    def group_iter(self) -> ty.Generator[int, None, None]:
        """Iterate over groups."""
        yield from self.groups.keys()


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
        # also, potentially remove it from a group
        if "groups" in image:
            if image["group_id"] != -1 and image["group_id"] in image["groups"]:
                if key in image["groups"][image["group_id"]]["keys"]:
                    index = image["groups"][image["group_id"]]["keys"].index(key)
                    image["groups"][image["group_id"]]["keys"].pop(index)
    return config
