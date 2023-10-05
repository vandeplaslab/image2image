"""Configuration."""
import typing as ty
from pathlib import Path

from koyo.typing import PathLike
from loguru import logger
from pydantic import BaseModel, Field, validator

from image2image._appdirs import USER_CONFIG_DIR
from image2image.enums import ViewerOrientation, ViewType


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

    # visuals
    theme: str = Field("light", title="Theme", description="Theme of the application.")

    # paths
    fixed_dir: str = Field("", title="Fixed directory", description="Directory with fixed images.")
    moving_dir: str = Field("", title="Moving directory", description="Directory with moving images.")
    output_dir: str = Field("", title="Output directory", description="Directory where output should be saved.")

    # telemetry
    telemetry_enabled: bool = Field(True, title="Enable telemetry", description="Enable telemetry.")
    telemetry_with_locals: bool = Field(True, title="Send locals", description="Send locals with telemetry.")

    @validator("fixed_dir", "moving_dir", "output_dir", pre=True, allow_reuse=True)
    def _validate_path(value: PathLike) -> str:
        """Validate path."""
        return str(value)

    @validator("viewer_orientation", pre=True, allow_reuse=True)
    def _validate_orientation(value: ty.Union[str, ViewerOrientation]) -> ViewerOrientation:
        """Validate path."""
        return ViewerOrientation(value)

    @validator("view_type", pre=True, allow_reuse=True)
    def _validate_view_type(value: ty.Union[str, ViewType]) -> ViewType:
        """Validate view_type."""
        return ViewType(value)

    @property
    def output_path(self) -> Path:
        """Get default output path."""
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        return USER_CONFIG_DIR / "config.json"

    def save(self) -> None:
        """Export configuration to file."""
        try:
            self.output_path.write_text(self.json(indent=4, exclude_unset=True))
            logger.info(f"Saved configuration to {self.output_path}")
        except Exception as e:
            logger.warning(f"Failed to save configuration to {self.output_path}: {e}")

    def load(self) -> None:
        """Load configuration from file."""
        from koyo.json import read_json_data

        if self.output_path.exists():
            try:
                data = read_json_data(self.output_path)
                for key, value in data.items():
                    if hasattr(self, key):
                        try:
                            setattr(self, key, value)
                        except Exception as e:
                            logger.warning(f"Failed to set {key}={value}: {e}")
                logger.info(f"Loaded configuration from {self.output_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {self.output_path}: {e}")


CONFIG: Config = Config()
