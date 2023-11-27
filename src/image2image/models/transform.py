"""Transform."""
from image2image_reader.models.transform import TransformData as _TransformData
from image2image_reader.models.transform import TransformModel
from koyo.typing import PathLike

__all__ = ("TransformData", "TransformModel")


class TransformData(_TransformData):
    """Transformation data."""

    @classmethod
    def from_i2r(cls, path: PathLike, validate_paths: bool = True) -> "TransformData":
        """Load directly from i2r."""
        from image2image.models.transformation import load_transform_from_file

        (
            transformation_type,
            _fixed_paths,
            _fixed_paths_missing,
            fixed_points,
            _moving_paths,
            _moving_paths_missing,
            moving_points,
            fixed_resolution,
            _moving_resolution,
        ) = load_transform_from_file(path, validate_paths=validate_paths)
        return TransformData(
            fixed_points=fixed_points,
            moving_points=moving_points,
            transformation_type=transformation_type,
            fixed_resolution=fixed_resolution,
        )
