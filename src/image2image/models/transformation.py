"""Transformation model."""
import typing as ty
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from skimage.transform import AffineTransform, ProjectiveTransform

from image2image.models.base import BaseModel
from image2image.models.data import DataModel
from image2image.models.utilities import _get_paths, _read_config_from_file

I2R_METADATA = ty.Tuple[
    str,
    ty.Optional[ty.List[Path]],
    ty.Optional[ty.List[Path]],
    ty.Optional[np.ndarray],
    ty.Optional[ty.List[Path]],
    ty.Optional[ty.List[Path]],
    ty.Optional[np.ndarray],
    float,
    float,
]

SCHEMA_VERSION: str = "1.3"


class Transformation(BaseModel):
    """Temporary object that holds transformation information."""

    # Transformation object
    transform: ty.Optional[ProjectiveTransform] = None
    # Type of transformation
    transformation_type: str = "affine"
    # Path to the image
    fixed_model: ty.Optional[DataModel] = None
    moving_model: ty.Optional[DataModel] = None
    # Date when the registration was created
    time_created: ty.Optional[datetime] = None
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None
    # affine model
    moving_initial_affine: ty.Optional[np.ndarray] = None

    @property
    def moving_to_fixed_ratio(self) -> float:
        """Ratio between the moving and fixed model."""
        if self.moving_model and self.fixed_model:
            return self.moving_model.min_resolution / self.fixed_model.min_resolution
        return 1.0

    @property
    def fixed_to_moving_ratio(self) -> float:
        """Ratio between the moving and fixed model."""
        if self.moving_model and self.fixed_model:
            try:
                return self.fixed_model.min_resolution / self.moving_model.min_resolution
            except ValueError:
                return 1.0
        return 1.0

    def is_valid(self) -> bool:
        """Returns True if the transformation is valid."""
        return self.transform is not None

    def is_recommended(self) -> tuple[bool, str]:
        """Returns True if the transformation is recommended."""
        if self.transform is None:
            return False, "No transformation found."
        elif self.moving_points is not None and len(self.moving_points) < 5:
            return False, "Too few moving points - we recommended at least 5 but the more the better."
        elif self.fixed_points is not None and len(self.fixed_points) < 5:
            return False, "Too few fixed points - we recommended at least 5 but the more the better."
        return True, ""

    def clear(self, clear_data: bool = True, clear_model: bool = True, clear_initial: bool = True) -> None:
        """Clear transformation."""
        self.transformation_type = "affine"
        if clear_data:
            self.transform = None
            self.time_created = None
            self.fixed_points = None
            self.moving_points = None
        if clear_initial:
            self.moving_initial_affine = None
        if clear_model:
            self.fixed_model = None
            self.moving_model = None

    def __call__(self, coords: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Transform coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        coords = np.asarray(coords)
        coords = coords.copy()
        if inverse:
            return self.inverse(coords)
        coords = self.transform(coords)
        return coords

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        """Inverse transformation of coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        coords = np.asarray(coords)
        coords = coords.copy()
        coords = self.transform.inverse(coords)
        return coords

    def error(self) -> float:
        """Return error of the transformation."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        moving_points = self.moving_points
        moving_points = self.apply_moving_initial_transform(moving_points)  # type: ignore[arg-type]
        transformed_points = self.transform(moving_points)
        return float(np.sqrt(np.sum((self.fixed_points - transformed_points) ** 2)))

    @property
    def matrix(self) -> np.ndarray:
        """Retrieve the transformation array."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform.params  # type: ignore[no-any-return]

    def apply_moving_initial_transform(self, coords: np.ndarray, inverse: bool = True) -> np.ndarray:
        """Transform coordinates.

        If transforming from moving to fixed, set inverse to True. Execute BEFORE any other transformations.
        If transforming from fixed to moving, set inverse to False. Execute AFTER any other transformations.
        """
        if self.moving_initial_affine is None:
            return coords
        transform = AffineTransform(matrix=self.moving_initial_affine)
        if inverse:
            return transform.inverse(coords)  # type: ignore[no-any-return]
        return transform(coords)  # type: ignore[no-any-return]

    def compute(self, yx: bool = True, px: bool = True) -> ProjectiveTransform:
        """Compute transformation matrix."""
        from image2image.utils.transform import compute_transform

        moving_points = self.moving_points
        fixed_points = self.fixed_points
        if moving_points is None or fixed_points is None:
            raise ValueError("No points found.")

        # apply moving initial affine
        moving_points = self.apply_moving_initial_transform(moving_points)

        # swap yx to xy
        if not yx:
            moving_points = moving_points[:, ::-1]
            fixed_points = fixed_points[:, ::-1]
        # swap px to um
        if not px:
            moving_points = moving_points * self.fixed_model.resolution  # type: ignore[union-attr]
            fixed_points = fixed_points * self.moving_model.resolution  # type: ignore[union-attr]

        transform = compute_transform(
            moving_points,  # source
            fixed_points,  # destination
            self.transformation_type,
        )
        # if self.moving_initial_affine is not None:
        #     transform = transform.params @ AffineTransform(matrix=self.moving_initial_affine)._inv_matrix
        #     transform = AffineTransform(matrix=transform)
        return transform

    def about(
        self,
        sep: str = "\n",
        error: bool = True,
        n: bool = True,
        split_by_dim: bool = False,
        transform: ty.Union[ProjectiveTransform, np.ndarray, None] = None,
    ) -> str:
        """Retrieve information about the model in textual format."""
        info = ""
        if self.transformation_type:
            info += f"type: {self.transformation_type}"
        if transform is not None:
            if isinstance(transform, np.ndarray):
                transform = AffineTransform(matrix=transform)
        else:
            transform = self.transform
        if transform:
            if hasattr(transform, "scale"):
                scale = transform.scale
                scale = (scale, scale) if isinstance(scale, float) else scale
                if split_by_dim:
                    info += f"{sep}scale(y): {scale[0]:.3f}"
                    info += f"{sep}scale(x): {scale[1]:.3f}"
                else:
                    info += f"{sep}scale: {scale[0]:.3f}, {scale[1]:.3f}"
            if hasattr(transform, "translation"):
                translation = transform.translation
                translation = (translation, translation) if isinstance(translation, float) else translation
                if split_by_dim:
                    info += f"{sep}translation(y): {translation[0]:.1f}"
                    info += f"{sep}translation(x): {translation[1]:.1f}"
                else:
                    info += f"{sep}translation: {translation[0]:.1f}, {translation[1]:.1f}"
            if hasattr(transform, "rotation"):
                rotation = transform.rotation
                info += f"{sep}rotation: {rotation:.3f}"
            if error:
                with suppress(ValueError):
                    info += f"{sep}error: {self.error():.2f}"
        if n:
            if self.fixed_points is not None:
                info += f"{sep}no. fixed: {len(self.fixed_points)}"
            if self.moving_points is not None:
                info += f"{sep}no. moving: {len(self.moving_points)}"
        return info

    def to_dict(self) -> ty.Dict:
        """Convert to dict."""
        fixed_mdl = self.fixed_model
        if not fixed_mdl:
            raise ValueError("No fixed model found.")
        moving_mdl = self.moving_model
        if not moving_mdl:
            raise ValueError("No moving model found.")
        if self.time_created is None:
            raise ValueError("No time_created found.")
        fixed_pts = self.fixed_points
        assert fixed_pts is not None, "No fixed points found."
        moving_pts = self.moving_points
        assert moving_pts is not None, "No moving points found."
        moving_pts = self.apply_moving_initial_transform(moving_pts)
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": "register",
            "time_created": self.time_created.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "fixed_points_yx_px": fixed_pts.tolist(),
            "fixed_points_yx_um": (fixed_pts * fixed_mdl.resolution).tolist(),
            "moving_points_yx_px": moving_pts.tolist(),
            "moving_points_yx_um": (moving_pts * moving_mdl.resolution).tolist(),
            "transformation_type": self.transformation_type,
            "fixed_paths": [
                {
                    "path": str(path),
                    "pixel_size_um": resolution,
                    "image_shape": tuple(map(int, image_shape)),
                }
                for (path, resolution, image_shape) in fixed_mdl.path_resolution_shape_iter()
            ],
            "moving_paths": [
                {
                    "path": str(path),
                    "pixel_size_um": resolution,
                    "image_shape": tuple(map(int, image_shape)),
                }
                for (path, resolution, image_shape) in moving_mdl.path_resolution_shape_iter()
            ],
            "initial_matrix_yx_um": self.moving_initial_affine.tolist()
            if self.moving_initial_affine is not None
            else [],
            "matrix_yx_px": self.compute(yx=True, px=True).params.tolist(),
            "matrix_yx_um": self.compute(yx=True, px=False).params.tolist(),
            "matrix_xy_px": self.compute(yx=False, px=True).params.tolist(),
            "matrix_xy_um": self.compute(yx=False, px=False).params.tolist(),
            "matrix_yx_px_inv": self.compute(yx=True, px=True)._inv_matrix.tolist(),
            "matrix_yx_um_inv": self.compute(yx=True, px=False)._inv_matrix.tolist(),
            "matrix_xy_px_inv": self.compute(yx=False, px=True)._inv_matrix.tolist(),
            "matrix_xy_um_inv": self.compute(yx=False, px=False)._inv_matrix.tolist(),
        }

    def to_file(self, path: PathLike) -> Path:
        """Export data as any supported format."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path)
        elif path.suffix == ".toml":
            self.to_toml(path)
        elif path.suffix == ".xml":
            self.to_xml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        logger.info(f"Exported to '{path}'")
        return path

    def to_xml(self, path: PathLike) -> None:
        """Export dat aas fusion file."""
        from image2image.utils.utilities import write_xml_registration

        affine = self.compute(yx=False, px=True).params
        affine = affine.flatten("F").reshape(3, 3)
        write_xml_registration(path, affine)


def load_transform_from_file(
    path: PathLike,
    fixed_image: bool = False,
    moving_image: bool = False,
    fixed_points: bool = True,
    moving_points: bool = True,
    validate_paths: bool = True,
) -> I2R_METADATA:
    """Load registration from file."""
    path = Path(path)
    data = _read_config_from_file(path)

    # image2image config
    if "schema_version" in data:
        (
            transformation_type,
            fixed_paths,
            fixed_missing_paths,
            _fixed_points,
            moving_paths,
            moving_missing_paths,
            _moving_points,
            fixed_resolution,
            moving_resolution,
        ) = _read_image2register_config(data, validate_paths)
    # imsmicrolink config
    elif "Project name" in data:
        (
            transformation_type,
            fixed_paths,
            fixed_missing_paths,
            _fixed_points,
            moving_paths,
            moving_missing_paths,
            _moving_points,
            fixed_resolution,
            moving_resolution,
        ) = _read_imsmicrolink_config(data, validate_paths)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}.")

    # apply config
    fixed_paths, fixed_missing_paths = (fixed_paths, fixed_missing_paths) if fixed_image else (None, None)
    moving_paths, moving_missing_paths = (moving_paths, moving_missing_paths) if moving_image else (None, None)
    _fixed_points = _fixed_points if fixed_points else None
    _moving_points = _moving_points if moving_points else None
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        _fixed_points,
        moving_paths,
        moving_missing_paths,
        _moving_points,
        fixed_resolution,
        moving_resolution,
    )


def _read_image2register_config(config: ty.Dict, validate_paths: bool = True) -> I2R_METADATA:
    """Read image2image configuration file."""
    schema_version = config["schema_version"]
    if schema_version == "1.0":
        return _read_image2register_v1_0_config(config, validate_paths)
    elif schema_version == "1.1":
        return _read_image2register_v1_1_config(config, validate_paths)
    # accept 1.2 and 1.3
    return _read_image2register_latest_config(config, validate_paths)


def _read_image2register_latest_config(config: ty.Dict, validate_paths: bool = True) -> I2R_METADATA:
    # read important fields
    fixed_paths, fixed_missing_paths, moving_paths, moving_missing_paths = [], [], [], []
    paths = [temp["path"] for temp in config["fixed_paths"]]
    if validate_paths:
        fixed_paths, fixed_missing_paths = _get_paths(paths)
    fixed_res = [temp["pixel_size_um"] for temp in config["fixed_paths"]]
    fixed_resolution = float(np.min(fixed_res))

    paths = [temp["path"] for temp in config["moving_paths"]]
    moving_res = [temp["pixel_size_um"] for temp in config["moving_paths"]]
    moving_resolution = float(np.min(moving_res))
    if validate_paths:
        moving_paths, moving_missing_paths = _get_paths(paths)
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
        fixed_resolution,
        moving_resolution,
    )


def _read_image2register_v1_1_config(config: ty.Dict, validate_paths: bool = True) -> I2R_METADATA:
    # read important fields
    fixed_paths, fixed_missing_paths, moving_paths, moving_missing_paths = [], [], [], []
    if validate_paths:
        fixed_paths, fixed_missing_paths = _get_paths(config["fixed_paths"])
        moving_paths, moving_missing_paths = _get_paths(config["moving_paths"])
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        1.0,
    )


def _read_image2register_v1_0_config(config: ty.Dict, validate_paths: bool = True) -> I2R_METADATA:
    # read important fields
    fixed_paths, fixed_missing_paths, moving_paths, moving_missing_paths = [], [], [], []
    if validate_paths:
        fixed_paths, fixed_missing_paths = _get_paths(config["micro_paths"])
        moving_paths, moving_missing_paths = _get_paths(config["ims_paths"])
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        1.0,
    )


def _read_imsmicrolink_config(config: ty.Dict, validate_paths: bool = True) -> I2R_METADATA:
    """Read imsmicrolink configuration file."""
    fixed_paths, fixed_missing_paths, moving_paths, moving_missing_paths = [], [], [], []
    if validate_paths:
        fixed_paths, fixed_missing_paths = _get_paths([config["PostIMS microscopy image"]])  # need to be a list
        moving_paths, moving_missing_paths = _get_paths(config["Pixel Map Datasets Files"])
    fixed_points = np.array(config["PAQ microscopy points (xy, px)"])[:, ::-1]
    moving_points = np.array(config["IMS pixel map points (xy, px)"])[:, ::-1]
    padding = config["padding"]
    if padding["x_left_padding (px)"]:
        moving_points[:, 1] -= padding["x_left_padding (px)"]
    if padding["y_top_padding (px)"]:
        moving_points[:, 0] -= padding["y_top_padding (px)"]
    transformation_type = "Affine"
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        1.0,
    )
