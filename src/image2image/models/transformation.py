"""Transformation model."""

import typing as ty
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from koyo.utilities import clean_path
from loguru import logger
from skimage.transform import AffineTransform, ProjectiveTransform

from image2image.models.base import BaseModel
from image2image.models.data import DataModel
from image2image.models.utilities import _get_paths, _read_config_from_file

I2R_METADATA = tuple[
    str,
    ty.Optional[list[Path]],
    ty.Optional[list[Path]],
    ty.Optional[np.ndarray],
    float,
    dict[str, dict],
    ty.Optional[list[Path]],
    ty.Optional[list[Path]],
    ty.Optional[np.ndarray],
    float,
    dict[str, dict],
]

SCHEMA_VERSION: str = "1.4"


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

    def is_error(self) -> tuple[bool, str]:
        """Returns True if the transformation is valid."""
        is_valid = True
        error = []
        if self.transform is None:
            is_valid = False
            error.append("No transformation found.")
        if self.fixed_model and not self.fixed_model.paths:
            is_valid = False
            error.append("No fixed image found (fixed path is not set) - <b>please load fixed image</b>.")
        if self.moving_model and not self.moving_model.paths:
            is_valid = False
            error.append("No moving image found (moving path is not set) - <b>please load moving image</b>.")
        return is_valid, "<br>".join(error)

    def is_recommended(self) -> tuple[bool, str]:
        """Returns True if the transformation is recommended."""
        recommended = True
        error = []
        if self.moving_points is not None and len(self.moving_points) < 5:
            recommended = False
            error.append(f"Too few moving points - we recommended at least 5 - you have {len(self.moving_points)}.")
        if self.fixed_points is not None and len(self.fixed_points) < 5:
            recommended = False
            error.append(f"Too few fixed points - we recommended at least 5 - you have {len(self.fixed_points)}.")
        return recommended, "<br>".join(error)

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
        transformed_points = transformed_points * self.fixed_model.get_resolution()  # type: ignore[union-attr]
        return float(np.sqrt(np.sum((self.fixed_points * self.fixed_model.get_resolution() - transformed_points) ** 2)))

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
            moving_points = moving_points * self.fixed_model.get_resolution()  # type: ignore[union-attr]
            fixed_points = fixed_points * self.moving_model.get_resolution()  # type: ignore[union-attr]

        return compute_transform(
            moving_points,  # source
            fixed_points,  # destination
            self.transformation_type,
        )

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
                    info += f"{sep}translation(y): {translation[0]:.0f} px"
                    info += f"{sep}translation(x): {translation[1]:.0f} px"
                else:
                    info += f"{sep}translation: {translation[0]:.1f}, {translation[1]:.1f}"
            if hasattr(transform, "rotation"):
                radians = transform.rotation
                degrees = radians * 180 / 3.141592653589793
                info += f"{sep}rotation: {radians:.3f} ({degrees:.3f}°)"
            if error:
                with suppress(ValueError):
                    info += f"{sep}error: {self.error():.2f}"
        if n:
            if self.fixed_points is not None:
                info += f"{sep}no. fixed: {len(self.fixed_points)}"
            if self.moving_points is not None:
                info += f"{sep}no. moving: {len(self.moving_points)}"
        return info

    def to_dict(self, moving_key: ty.Optional[str] = None) -> dict:
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
        fixed_paths = [
            {
                "path": str(path),
                "pixel_size_um": resolution,
                "image_shape": tuple(map(int, image_shape)),
                "reader_kws": reader_kws,
            }
            for (_, path, resolution, image_shape, reader_kws) in fixed_mdl.path_resolution_shape_iter()
        ]
        moving_paths = []
        for key, path, resolution, image_shape, reader_kws in moving_mdl.path_resolution_shape_iter():
            if moving_key and key != moving_key:
                continue
            moving_paths.append(
                {
                    "path": str(path),
                    "pixel_size_um": resolution,
                    "image_shape": tuple(map(int, image_shape)),
                    "reader_kws": reader_kws,
                }
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "tool": "register",
            "time_created": self.time_created.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "transformation_type": self.transformation_type,
            "fixed_paths": fixed_paths,
            "moving_paths": moving_paths,
            "initial_matrix_yx_um": self.moving_initial_affine.tolist()
            if self.moving_initial_affine is not None
            else [],
            "fixed_points_yx_px": fixed_pts.tolist(),
            "fixed_points_yx_um": (fixed_pts * fixed_mdl.get_resolution()).tolist(),
            "moving_points_yx_px": moving_pts.tolist(),
            "moving_points_yx_um": (moving_pts * moving_mdl.get_resolution()).tolist(),
            "matrix_yx_px": self.compute(yx=True, px=True).params.tolist(),
            "matrix_yx_um": self.compute(yx=True, px=False).params.tolist(),
            "matrix_xy_px": self.compute(yx=False, px=True).params.tolist(),
            "matrix_xy_um": self.compute(yx=False, px=False).params.tolist(),
            "matrix_yx_px_inv": self.compute(yx=True, px=True)._inv_matrix.tolist(),
            "matrix_yx_um_inv": self.compute(yx=True, px=False)._inv_matrix.tolist(),
            "matrix_xy_px_inv": self.compute(yx=False, px=True)._inv_matrix.tolist(),
            "matrix_xy_um_inv": self.compute(yx=False, px=False)._inv_matrix.tolist(),
        }

    def to_file(self, path: PathLike, moving_key: ty.Optional[PathLike] = None) -> Path:
        """Export data as any supported format."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path, moving_key=moving_key)
        elif path.suffix == ".toml":
            self.to_toml(path, moving_key=moving_key)
        elif path.suffix == ".xml":
            self.to_xml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        logger.info(f"Exported to '{path}'")
        return path

    def to_xml(self, path: PathLike) -> None:
        """Export data as fusion file."""
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
            fixed_resolution,
            fixed_reader_kws,
            moving_paths,
            moving_missing_paths,
            _moving_points,
            moving_resolution,
            moving_reader_kws,
        ) = _read_image2register_config(data, validate_paths)
    # imsmicrolink config
    elif "Project name" in data:
        (
            transformation_type,
            fixed_paths,
            fixed_missing_paths,
            _fixed_points,
            fixed_resolution,
            fixed_reader_kws,
            moving_paths,
            moving_missing_paths,
            _moving_points,
            moving_resolution,
            moving_reader_kws,
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
        fixed_resolution,
        fixed_reader_kws,
        moving_paths,
        moving_missing_paths,
        _moving_points,
        moving_resolution,
        moving_reader_kws,
    )


def _read_image2register_config(config: dict, validate_paths: bool = True) -> I2R_METADATA:
    """Read image2image configuration file."""
    schema_version = config["schema_version"]
    if schema_version == "1.0":
        return _read_image2register_v1_0_config(config, validate_paths)
    if schema_version == "1.1":
        return _read_image2register_v1_1_config(config, validate_paths)
    # accept 1.2 and 1.3
    return _read_image2register_latest_config(config, validate_paths)


def _read_image2register_latest_config(config: dict, validate_paths: bool = True) -> I2R_METADATA:
    # read important fields
    fixed_paths, fixed_missing_paths, fixed_reader_kws = [], [], {}
    if validate_paths:
        paths = [clean_path(temp["path"]) for temp in config["fixed_paths"]]
        fixed_reader_kws = {
            Path(clean_path(temp["path"])).name: temp.get("reader_kws", None) for temp in config["fixed_paths"]
        }
        fixed_paths, fixed_missing_paths = _get_paths(paths)
    fixed_res = [temp["pixel_size_um"] for temp in config["fixed_paths"]]
    fixed_resolution = float(np.min(fixed_res)) if fixed_res else 0.0

    moving_paths, moving_missing_paths, moving_reader_kws = [], [], {}
    if validate_paths:
        paths = [clean_path(temp["path"]) for temp in config["moving_paths"]]
        moving_reader_kws = {
            Path(clean_path(temp["path"])).name: temp.get("reader_kws", None) for temp in config["moving_paths"]
        }
        moving_paths, moving_missing_paths = _get_paths(paths)
    moving_res = [temp["pixel_size_um"] for temp in config["moving_paths"]]
    moving_resolution = float(np.min(moving_res)) if moving_res else 0.0

    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        fixed_resolution,
        fixed_reader_kws,
        moving_paths,
        moving_missing_paths,
        moving_points,
        moving_resolution,
        moving_reader_kws,
    )


def _read_image2register_v1_1_config(config: dict, validate_paths: bool = True) -> I2R_METADATA:
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
        1.0,
        {},
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        {},
    )


def _read_image2register_v1_0_config(config: dict, validate_paths: bool = True) -> I2R_METADATA:
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
        1.0,
        {},
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        {},
    )


def _read_imsmicrolink_config(config: dict, validate_paths: bool = True) -> I2R_METADATA:
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
        1.0,
        {},
        moving_paths,
        moving_missing_paths,
        moving_points,
        1.0,
        {},
    )
