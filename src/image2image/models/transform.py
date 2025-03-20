"""Transform."""

from image2image_io.models.transform import TransformData as _TransformData
from image2image_io.models.transform import TransformModel
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
            fixed_resolution,
            _fixed_reader_kws,
            _moving_paths,
            _moving_paths_missing,
            moving_points,
            moving_resolution,
            _moving_reader_kws,
        ) = load_transform_from_file(path, validate_paths=validate_paths)
        return cls(
            fixed_points=fixed_points,
            moving_points=moving_points,
            transformation_type=transformation_type,
            fixed_resolution=fixed_resolution,
            moving_resolution=moving_resolution,
        )

    @classmethod
    def recalculate(cls, path: PathLike, with_backup: bool = True) -> None:
        """Recalculate and export."""
        from koyo.json import read_json, write_json

        # initialize model
        obj = cls.from_i2r(path, validate_paths=False)

        # create backup in case something goes wrong
        if with_backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if not backup_path.exists():
                backup_path.write_text(path.read_text())

        # load configuration
        config = read_json(path)
        config["matrix_yx_px"] = obj.compute(yx=True, px=True).params.tolist()
        config["matrix_yx_um"] = obj.compute(yx=True, px=False).params.tolist()
        config["matrix_xy_px"] = obj.compute(yx=False, px=True).params.tolist()
        config["matrix_xy_um"] = obj.compute(yx=False, px=False).params.tolist()
        config["matrix_yx_px_inv"] = obj.compute(yx=True, px=True)._inv_matrix.tolist()
        config["matrix_yx_um_inv"] = obj.compute(yx=True, px=False)._inv_matrix.tolist()
        config["matrix_xy_px_inv"] = obj.compute(yx=False, px=True)._inv_matrix.tolist()
        config["matrix_xy_um_inv"] = obj.compute(yx=False, px=False)._inv_matrix.tolist()

        # export
        write_json(path, config)
