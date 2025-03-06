"""Three-D model."""

import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from pydantic import Field, field_validator

from image2image.config import get_elastix3d_config
from image2image.models.base import BaseModel
from image2image.models.utilities import _get_paths, _read_config_from_file
from image2image.utils.transform import combined_transform

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from image2image_reg.workflows import ElastixReg


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
    channel_ids: list = Field(default_factory=list, title="Channel ids")
    metadata: ty.Dict = Field(default_factory=dict, title="Metadata")
    mask_bbox: ty.Optional[tuple[int, int, int, int]] = Field(None, title="Mask bounding box")
    mask_polygon: ty.Optional[np.ndarray] = Field(None, title="Mask polygon")

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
            self.rotate += get_elastix3d_config().rotate_step_size
        else:
            self.rotate -= get_elastix3d_config().rotate_step_size
        if self.rotate > 360:
            self.rotate -= 360
        elif self.rotate < 0:
            self.rotate += 360
        if self.rotate == 360:
            self.rotate = 0

    def apply_translate(self, which: str) -> None:
        """Apply rotation."""
        if which == "up":
            self.translate_y -= get_elastix3d_config().translate_step_size
        elif which == "down":
            self.translate_y += get_elastix3d_config().translate_step_size
        elif which == "left":
            self.translate_x -= get_elastix3d_config().translate_step_size
        else:
            self.translate_x += get_elastix3d_config().translate_step_size

    def affine(
        self, shape: tuple[int, int], scale: ty.Optional[tuple[float, float]] = None, only_translate: bool = False
    ) -> np.ndarray:
        """Calculate affine transformation."""
        if scale is None:
            scale = self.scale
        return combined_transform(
            shape,
            scale,
            0 if only_translate else self.rotate,
            (self.translate_y, self.translate_x),
            False if only_translate else self.flip_lr,
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
            "metadata": self.metadata,
        }


class RegistrationGroup(BaseModel):
    """Registration group."""

    keys: list[str] = Field(default_factory=list, title="Keys")
    group_id: int = Field(-1, title="Group ID")
    mask_bbox: ty.Optional[tuple[int, int, int, int]] = Field(None, title="Mask bounding box")
    mask_polygon: ty.Optional[np.ndarray] = Field(None, title="Mask polygon")

    @field_validator("mask_polygon", mode="before")
    def _validate_polygon(cls, v) -> ty.Optional[np.ndarray]:
        if isinstance(v, list):
            return np.asarray(v)
        if isinstance(v, np.ndarray):
            return v
        return None

    def is_masked(self) -> bool:
        """Check whether mask is present."""
        return self.mask_bbox is not None or self.mask_polygon is not None

    def has_metadata(self, registration: "Registration") -> bool:
        """Check whether metadata is present."""
        return all(registration.images[key].metadata for key in self.keys)

    def clear(self) -> None:
        """Remove existing keys."""
        self.keys = []

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        return {
            "keys": self.keys,
            "group_id": self.group_id,
            "mask_bbox": self.mask_bbox,
            "mask_polygon": self.mask_polygon,
        }

    def get_mask_bbox(
        self, image: "BaseReader", affine: ty.Optional[np.ndarray] = None
    ) -> ty.Optional[tuple[int, int, int, int]]:
        """Get mask bounding box."""
        if self.mask_bbox:
            bbox = np.asarray(self.mask_bbox)
            # if affine is not None:
            #     bbox = np.dot(affine[:2, :2], bbox.T).T + affine[:2, 2]
            # convert to pixel coordinates and round
            return tuple(np.round(bbox * image.inv_resolution).astype(int))
        return None

    def get_mask_polygon(self, image: "BaseReader", affine: ty.Optional[np.ndarray] = None) -> ty.Optional[np.ndarray]:
        """Get mask polygon."""
        if self.mask_polygon is not None:
            yx = self.mask_polygon
            if affine is not None:
                yx = np.dot(affine[:2, :2], yx.T).T + affine[:2, 2]
            # convert to pixel coordinates and round
            yx = yx * image.inv_resolution
            return np.round(yx).astype(np.int32)[:, ::-1]
        return None

    def to_iwsireg(
        self,
        registration: "Registration",
        output_dir: PathLike,
        name: str,
        transformations: tuple[str, ...] = ("rigid", "affine"),
        prefix: str = "c",
        index_mode: str = "auto",
        export_mode: str = "all",
        first_only: bool = False,
        direct: bool = False,
        as_uint8: bool = False,
    ) -> Path:
        """Generate iwsireg configuration file."""
        from image2image_io.readers import get_simple_reader
        from image2image_reg.enums import CoordinateFlip
        from image2image_reg.models import Preprocessing
        from image2image_reg.workflows import ElastixReg

        if export_mode == "all":
            export_mode = "Export with mask + affine initialization"

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
        # if len(index_to_image) == 1:
        #     raise ValueError("Cannot register a single image")

        # create ElastixReg object
        path = Path(output_dir) / f"{name}.wsireg"
        obj = ElastixReg(name=name, output_dir=output_dir, cache=True, merge=True)
        for index, key in index_to_image.items():
            image = registration.images[key]
            name_index = index if index_mode == "auto" else image.metadata[index_mode]
            reader = get_simple_reader(image.path)
            pre = Preprocessing.brightfield() if reader.is_rgb else Preprocessing.fluorescence()

            affine = image.affine(reader.image_shape, reader.scale)
            if RegistrationImage.is_identity(affine):
                affine = None

            # use affine matrix if present and user explicitly requested it
            if "affine initialization" in export_mode and affine is not None:
                pre.affine = affine
            # affine matrix without rotation/flip
            elif "affine(translate)" in export_mode:
                affine = image.affine(reader.image_shape, reader.scale, only_translate=True)
                if not RegistrationImage.is_identity(affine):
                    pre.affine = affine
                pre.rotate_counter_clockwise = 360 - image.rotate  # we store it as clockwise...
                # pre.rotate_counter_clockwise = image.rotate  # we store it as clockwise...
                if image.flip_lr:
                    pre.flip = CoordinateFlip.HORIZONTAL
            # use translation/rotation/flip if present and user explicitly requested it
            elif "translation/rotation/flip" in export_mode:
                pre.rotate_counter_clockwise = 360 - image.rotate  # we store it as clockwise...
                # pre.rotate_counter_clockwise = image.rotate  # we store it as clockwise...
                if image.flip_lr:
                    pre.flip = CoordinateFlip.HORIZONTAL
                if image.translate_x:
                    pre.translate_x = int(image.translate_x)
                if image.translate_y:
                    pre.translate_y = int(image.translate_y)
            if first_only:
                pre.channel_indices = [0]
                pre.channel_names = [reader.channel_names[0]]
            else:
                pre.channel_indices = reader.channel_ids
                pre.channel_names = reader.channel_names

            kws: dict[str, ty.Any] = {}
            if "with mask" in export_mode:
                inv_affine = np.linalg.inv(affine) if affine is not None else None
                # if is_reference:
                kws["mask_bbox"] = self.get_mask_bbox(reader, inv_affine)
                kws["mask_polygon"] = self.get_mask_polygon(reader, inv_affine)
                kws["transform_mask"] = True
            if as_uint8:
                kws["export"] = {"as_uint8": True}
            # add modality to the project
            obj.add_modality(
                f"{prefix}{name_index}", image.path, pixel_size=reader.resolution, preprocessing=pre, **kws
            )

        # add registration paths
        if kind == "cascade":
            self._add_cascade_paths(
                obj,
                registration,
                reference,
                index_to_image,
                transformations=transformations,
                prefix=prefix,
                index_mode=index_mode,
                direct=direct,
            )
        elif kind == "converge":
            if not reference:
                raise ValueError("Reference must be present to converge")
            self._add_converge_paths(
                obj,
                registration,
                reference,
                index_to_image,
                transformations=transformations,
                prefix=prefix,
                index_mode=index_mode,
                direct=direct,
            )
        # validate the registration
        if not obj.validate(allow_not_registered=True)[0]:
            raise ValueError("Invalid registration")
        # save the registration
        obj.save()
        return path

    def to_preview(
        self, registration: "Registration", prefix: str = "c", index_mode: str = "auto", target_mode: str = "sequential"
    ) -> str:
        """Generate preview of the registration paths."""
        # retrieve images
        images = [registration.images[key] for key in self.keys]

        # sort images according to the image order, and if one is not specified, use the key
        if all(image.image_order == 0 for image in images):
            for index, image in enumerate(natsorted(images, key=lambda x: x.key)):
                image.image_order = index

        index_to_image, kind, reference = self.sort(registration)
        if len(index_to_image) == 1:
            return "<no preview available>"

        if kind == "cascade" and target_mode != "next":
            return self._preview_cascade_paths(
                registration,
                index_to_image,
                reference,
                prefix=prefix,
                index_mode=index_mode,
                direct=target_mode == "reference",
            )[0]
        if kind == "converge" and target_mode != "next":
            if not reference:
                raise ValueError("Reference must be present to converge")
            return self._preview_converge_paths(
                registration,
                index_to_image,
                reference,
                prefix=prefix,
                index_mode=index_mode,
                direct=target_mode == "reference",
            )[0]
        # else:
        #     return self._preview_next_paths(
        #         registration, index_to_image, reference, prefix=prefix, index_mode=index_mode,
        #     )
        return "<no preview available>"

    def _add_cascade_paths(
        self,
        obj: "ElastixReg",
        registration: "Registration",
        reference: ty.Optional[str],
        index_to_image: dict[int, str],
        transformations: tuple[str, ...] = ("rigid", "affine"),
        prefix: str = "c",
        index_mode: str = "auto",
        direct: bool = False,
    ) -> None:
        """Add cascade paths."""
        if reference is None:
            reference = index_to_image[0]
        _, paths = self._preview_cascade_paths(
            registration, index_to_image, reference, prefix=prefix, index_mode=index_mode, direct=direct
        )
        for source, target, through in paths:
            obj.add_registration_path(source, target, transform=transformations, through=through)
            logger.trace(f"Added registration path {source} -> {target} via {through} ({transformations})")

    def _add_converge_paths(
        self,
        obj: "ElastixReg",
        registration: "Registration",
        reference: str,
        index_to_image: dict[int, str],
        transformations: tuple[str, ...] = ("rigid", "affine"),
        prefix: str = "c",
        index_mode: str = "auto",
        direct: bool = False,
    ) -> None:
        """Add cascade paths."""
        _, paths = self._preview_converge_paths(
            registration, index_to_image, reference, prefix=prefix, index_mode=index_mode, direct=direct
        )
        for source, target, through in paths:
            obj.add_registration_path(source, target, transform=transformations, through=through)
            logger.trace(f"Added registration path {source} -> {target} via {through} ({transformations})")

    @staticmethod
    def _preview_cascade_paths(
        registration: "Registration",
        index_to_image: dict[int, str],
        reference: ty.Optional[str],
        prefix: str = "c",
        index_mode: str = "auto",
        direct: bool = False,
    ) -> ty.Tuple[str, list[tuple[str, str, ty.Optional[str]]]]:
        """Generate preview of the registration paths."""
        ref_image = registration.images[reference] if reference else None
        ref_name_index = 0
        if ref_image:
            ref_name_index = ref_image.metadata[index_mode] if index_mode != "auto" else 0

        lines = [f"<b>Reference: {reference} ({f'{prefix}{ref_name_index}' if reference else 'None'})</b><br>"]
        paths: list[tuple[str, str, ty.Optional[str]]] = []
        for image_index, key in index_to_image.items():
            image = registration.images[key]
            name_index = image_index if index_mode == "auto" else image.metadata[index_mode]
            # no need to register the first image, we will register against it
            if image_index == 0:
                continue
            # register second image to the first - no need for through modality
            if image_index == 1 or direct:
                lines.append(
                    f"- {key} -> {reference}<br><b>({prefix}{name_index}"
                    f" -> {f'{prefix}{ref_name_index}' if reference else 'None'})</b>"
                )
                paths.append((f"{prefix}{name_index}", f"{prefix}{ref_name_index}", None))
            else:
                lines.append(
                    f"- {key} -> {reference} via {index_to_image[image_index - 1]}"
                    f"<br><b>({prefix}{image_index} -> {f'{prefix}{ref_name_index}' if reference else 'None'} via"
                    f" {prefix}{name_index - 1})</b>"
                )
                paths.append(
                    (
                        f"{prefix}{name_index}",
                        f"{prefix}{ref_name_index}",
                        f"{prefix}{name_index - 1}",
                    )
                )
        return "<br>".join(lines), paths

    @staticmethod
    def _preview_converge_paths(
        registration: "Registration",
        index_to_image: dict[int, str],
        reference: str,
        prefix: str = "c",
        index_mode: str = "auto",
        direct: bool = False,
    ) -> ty.Tuple[str, list[tuple[str, str, ty.Optional[str]]]]:
        """Preview converge paths."""
        # find the index of reference in the index_to_image
        image_orders = list(index_to_image.keys())
        index = list(index_to_image.values()).index(reference)
        ref_index = image_orders[index]
        image_orders_before = image_orders[:index]
        image_orders_after = image_orders[index + 1 :]

        ref_image = registration.images[reference] if reference else None
        ref_name_index = -1
        if ref_image:
            ref_name_index = ref_image.metadata[index_mode] if index_mode != "auto" else ref_image.image_order

        lines = [f"<b>Reference: {reference} ({prefix}{ref_name_index})</b><br>"]
        paths: list[tuple[str, str, ty.Optional[str]]] = []
        # add images before the reference
        n = len(image_orders_before)
        for index, image_index in enumerate(image_orders_before):
            if image_index == ref_index:
                break
            image = registration.images[index_to_image[image_index]]
            name_index = image_index if index_mode == "auto" else image.metadata[index_mode]
            if index == n - 1 or direct:
                # from to
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference}<br><b>({prefix}{name_index}"
                    f" -> {prefix}{ref_name_index})</b>"
                )
                paths.append((f"{prefix}{name_index}", f"{prefix}{ref_name_index}", None))
            else:
                image = registration.images[index_to_image[image_index + 1]]
                via_name_index = image_orders_before[index + 1] if index_mode == "auto" else image.metadata[index_mode]
                # from to via
                lines.append(
                    f"- {index_to_image[image_index]} -> {reference} via {index_to_image[image_index + 1]}"
                    f"<br><b>({prefix}{name_index} -> {prefix}{ref_name_index} via {prefix}{via_name_index})</b>"
                )
                paths.append((f"{prefix}{name_index}", f"{prefix}{ref_name_index}", f"{prefix}{via_name_index}"))

        # add images after the reference
        n = len(image_orders_after)
        lines_after = []
        paths_after = []
        for index, image_index in enumerate(reversed(image_orders_after)):
            if image_index == ref_index:
                break
            image = registration.images[index_to_image[image_index]]
            name_index = image_index if index_mode == "auto" else image.metadata[index_mode]
            if index == n - 1 or direct:
                # from to
                lines_after.append(
                    f"- {index_to_image[image_index]} -> {reference}<br><b>({prefix}{name_index}"
                    f" -> {prefix}{ref_name_index})</b>"
                )
                paths_after.append((f"{prefix}{name_index}", f"{prefix}{ref_name_index}", None))
            else:
                image = registration.images[index_to_image[image_index - 1]]
                via_name_index = image_orders_after[index - 1] if index_mode == "auto" else image.metadata[index_mode]
                # from -> to -> via
                lines_after.append(
                    f"- {index_to_image[image_index]} -> {reference} via {index_to_image[image_index - 1]}"
                    f"<br><b>({prefix}{name_index} -> {prefix}{ref_name_index} via {prefix}{via_name_index})</b>"
                )
                paths_after.append((f"{prefix}{name_index}", f"{prefix}{ref_name_index}", f"{prefix}{via_name_index}"))
        lines.extend(reversed(lines_after))
        paths.extend(reversed(paths_after))
        return "<br>".join(lines), paths

    def sort(self, registration: "Registration", reset: bool = True) -> tuple[dict[int, str], str, ty.Optional[str]]:
        """Sort keys according to the following rules.

        - If no reference is present:
            - sort by the image_order if present
            - otherwise natural sorting by names
        - If reference is present, and it's the first image:
            - reference must be first
            - then sort by image_order if present
            - otherwise natural sorting by names
        - If reference is present, but it's not the first image:
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

        # if reference is present, but it's not the first image, reference does not have to be first, all images must
        # be pointed TO the reference, sort by image_order if present, otherwise natural sorting by names
        # e.g. if we had the following images 1, 2, ref, 4, 5, then we would have 1 -> 2 -> ref <- 4 <- 5
        # we need to sort the keys so that the reference is in the middle
        if ref_index != 0:
            kind = "converge"

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

    def get_metadata_keys(self) -> list[str]:
        """Return list of keys associated with EACH image."""
        keys: list[str] = []
        for image in self.images.values():
            if image.metadata:
                keys.extend(image.metadata.keys())
        keys = list(set(keys))
        # make sure to only return keys that are present in EACH image
        keys_ = deepcopy(keys)
        for key in keys_:
            if not all(key in image.metadata for image in self.images.values()):
                keys.remove(key)
        return keys

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

    def key_iter(self) -> ty.Generator[str, None, None]:
        """Iterate over keys according to the `image_order` attribute."""
        order = sorted(self.images, key=lambda key: self.images[key].image_order)
        yield from order

    def group_iter(self) -> ty.Generator[int, None, None]:
        """Iterate over groups."""
        yield from self.groups.keys()

    def is_single_project(self) -> bool:
        """Check whether each group contains only one image."""
        return all(len(group.keys) == 1 for group in self.groups.values())

    def to_iwsireg(
        self,
        output_dir: PathLike,
        name: str,
        transformations: tuple[str, ...] = ("rigid", "affine"),
        prefix: str = "c",
        index_mode: str = "auto",
        export_mode: str = "all",
        first_only: bool = False,
        direct: bool = False,
        as_uint8: bool = False,
    ) -> Path:
        """Generate iwsireg configuration file."""
        from image2image_io.readers import get_simple_reader
        from image2image_reg.enums import CoordinateFlip
        from image2image_reg.models import Preprocessing
        from image2image_reg.workflows import ElastixReg

        if export_mode == "all":
            export_mode = "Export with mask + affine initialization"

        # create iwsireg object
        path = Path(output_dir) / f"{name}.wsireg"
        obj = ElastixReg(name=name, output_dir=output_dir, cache=True, merge=True)

        for group_index, group in enumerate(self.groups.values()):
            # retrieve images
            images = [self.images[key] for key in group.keys]
            for image in images:
                if not image.path:
                    continue
                assert image.path.exists(), f"Path does not exist: {image.path}"

            # sort images according to the image order, and if one is not specified, use the key
            if all(image.image_order == 0 for image in images):
                for index, image in enumerate(natsorted(images, key=lambda x: x.key)):
                    image.image_order = index

            index_to_image, kind, reference = group.sort(self)
            for index, key in index_to_image.items():
                image = self.images[key]
                name_index = index if index_mode == "auto" else image.metadata[index_mode]
                reader = get_simple_reader(image.path)
                pre = Preprocessing.brightfield() if reader.is_rgb else Preprocessing.fluorescence()

                affine: ty.Optional[np.ndarray] = image.affine(reader.image_shape, reader.scale)
                if RegistrationImage.is_identity(affine):
                    affine = None

                # use affine matrix if present and user explicitly requested it
                if "affine initialization" in export_mode and affine is not None:
                    pre.affine = affine
                # affine matrix without rotation/flip
                elif "affine(translate)" in export_mode:
                    affine = image.affine(reader.image_shape, reader.scale, only_translate=True)
                    if not RegistrationImage.is_identity(affine):
                        pre.affine = affine
                    pre.rotate_counter_clockwise = 360 - image.rotate  # we store it as clockwise...
                    # pre.rotate_counter_clockwise = image.rotate  # we store it as clockwise...
                    if image.flip_lr:
                        pre.flip = CoordinateFlip.HORIZONTAL
                # use translation/rotation/flip if present and user explicitly requested it
                elif "translation/rotation/flip" in export_mode:
                    pre.rotate_counter_clockwise = 360 - image.rotate  # we store it as clockwise...
                    # pre.rotate_counter_clockwise = image.rotate  # we store it as clockwise...
                    if image.flip_lr:
                        pre.flip = CoordinateFlip.HORIZONTAL
                    if image.translate_x:
                        pre.translate_x = int(image.translate_x)
                    if image.translate_y:
                        pre.translate_y = int(image.translate_y)
                if first_only:
                    pre.channel_indices = [0]
                    pre.channel_names = [reader.channel_names[0]]
                else:
                    pre.channel_indices = reader.channel_ids
                    pre.channel_names = reader.channel_names

                kws: dict[str, ty.Any] = {}
                if "with mask" in export_mode:
                    inv_affine = np.linalg.inv(affine) if affine is not None else None
                    # if is_reference:
                    kws["mask_bbox"] = group.get_mask_bbox(reader, inv_affine)
                    kws["mask_polygon"] = group.get_mask_polygon(reader, inv_affine)
                    kws["transform_mask"] = True
                if as_uint8:
                    kws["export"] = {"as_uint8": True}
                # add modality to the project
                obj.add_modality(
                    f"{group_index}-{prefix}{name_index}",
                    image.path,
                    pixel_size=reader.resolution,
                    preprocessing=pre,
                    **kws,
                )

        # validate the registration
        if not obj.validate(allow_not_registered=True)[0]:
            raise ValueError("Invalid registration")
        # save the registration
        obj.save()
        return path


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
        if "groups" in image and image["group_id"] != -1 and image["group_id"] in image["groups"]:
            if key in image["groups"][image["group_id"]]["keys"]:
                index = image["groups"][image["group_id"]]["keys"].index(key)
                image["groups"][image["group_id"]]["keys"].pop(index)
    return config
