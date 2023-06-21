"""Configuration."""
from pathlib import Path

from qtextra.config.config import ConfigBase

from ims2micro.appdirs import USER_CONFIG_DIR


class Config(ConfigBase):
    """Configuration of few parameters."""

    # view parameters
    opacity_fixed: int = 100
    opacity_moving: int = 75
    size_fixed: int = 3
    size_moving: int = 1
    label_size: int = 12
    label_color: str = "#FFFF00"
    viewer_orientation: str = "vertical"
    show_transformed: bool = True

    # paths
    microscopy_dir: str = ""
    imaging_dir: str = ""
    output_dir: str = ""

    @property
    def output_path(self) -> Path:
        """Get default output path."""
        return USER_CONFIG_DIR / self.DEFAULT_CONFIG_NAME


CONFIG = Config(None)
