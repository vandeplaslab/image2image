"""Configuration."""
from pathlib import Path

from koyo.typing import PathLike
from pydantic import Field, validator, BaseModel
from ims2micro.enums import ViewerOrientation, ViewType
from loguru import logger
from ims2micro.appdirs import USER_CONFIG_DIR
import typing as ty


class Config(BaseModel):
    """Configuration of few parameters."""

    # view parameters
    opacity_fixed: int = Field(
        100, ge=0, le=100, step_size=10, title="Opacity (fixed)", description="Opacity of the fixed image"
    )
    opacity_moving: int = Field(
        75, ge=0, le=100, step_size=10, title="Opacity (moving)", description="Opacity of the moving image"
    )
    size_fixed: int = Field(
        3, ge=1, le=40, step_size=1, title="Size (fixed)", description="Size of the points shown in the fixed image."
    )
    size_moving: int = Field(
        10, ge=1, le=40, step_size=1, title="Size (moving)", description="Size of the points shown in the moving image."
    )
    label_size: int = Field(
        12, ge=4, le=60, step_size=4, title="Label size", description="Size of the text associated with each label."
    )
    label_color: str = Field(
        "#FFFF00", title="Label color", description="Color of the text associated with each label."
    )
    viewer_orientation: ViewerOrientation = Field(
        ViewerOrientation.VERTICAL, title="Viewer orientation", description="Orientation of the viewer."
    )
    view_type: ViewType = Field(ViewType.RANDOM, title="View type", description="IMS view type.")
    show_transformed: bool = Field(
        True, title="Show transformed", description="If checked, transformed moving image will be shown."
    )

    # paths
    microscopy_dir: str = Field("", title="Microscopy directory", description="Directory with microscopy images.")
    imaging_dir: str = Field("", title="Imaging directory", description="Directory with imaging images.")
    output_dir: str = Field("", title="Output directory", description="Directory where output will be saved.")

    @validator("microscopy_dir", "imaging_dir", "output_dir", pre=True, allow_reuse=True)
    def _validate_path(value: PathLike) -> str:
        """Validate path."""
        return str(value)

    @validator("viewer_orientation", pre=True, allow_reuse=True)
    def _validate_orientation(value: ty.Union[str, ViewerOrientation]) -> ViewerOrientation:
        """Validate path."""
        return ViewerOrientation(value)

    @property
    def output_path(self) -> Path:
        """Get default output path."""
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return USER_CONFIG_DIR / "config.json"

    def save(self):
        """Export configuration to file."""
        self.output_path.write_text(self.json(indent=4, exclude_unset=True))

    def load(self):
        """Load configuration from file."""
        from koyo.json import read_json_data

        if self.output_path.exists():
            try:
                data = read_json_data(self.output_path)
                self.__dict__.update(data)
                logger.info(f"Loaded configuration from {self.output_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.output_path}: {e}")


CONFIG = Config()
