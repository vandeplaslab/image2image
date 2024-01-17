"""Configuration."""
import typing as ty

from koyo.config import BaseConfig
from koyo.system import IS_MAC, IS_PYINSTALLER, is_envvar
from koyo.typing import PathLike
from pydantic import Field, validator

from image2image.enums import ViewerOrientation
from image2image.utils._appdirs import USER_CONFIG_DIR


class Config(BaseConfig):
    """Configuration of few parameters."""

    USER_CONFIG_DIR = USER_CONFIG_DIR

    # paths
    output_dir: str = Field(
        "", title="Output directory", description="Directory where output should be saved.", in_app=False
    )

    # visuals
    theme: str = Field(
        "light", title="Theme", description="Theme of the application.", options=["light", "dark"], in_app=True
    )

    # Crop-app parameters
    first_time_crop: bool = Field(True, title="First time", description="First time running the crop app.", in_app=True)
    confirm_close_crop: bool = Field(True, title="Confirm close", description="Confirm close crop app.", in_app=True)

    # Register-app parameters
    sync_views: bool = Field(True, title="Sync views", description="Sync views.", in_app=False)
    zoom_factor: float = Field(
        7.5, ge=1, le=10.0, step_size=0.25, n_decimals=2, title="Zoom factor", description="Zoom factor."
    )
    opacity_fixed: int = Field(
        100, ge=0, le=100, step_size=10, title="Opacity (fixed)", description="Opacity of the fixed image", in_app=False
    )
    opacity_moving: int = Field(
        75,
        ge=0,
        le=100,
        step_size=10,
        title="Opacity (moving)",
        description="Opacity of the moving image",
        in_app=False,
    )
    size_fixed: int = Field(
        3,
        ge=1,
        le=40,
        step_size=1,
        title="Size (fixed)",
        description="Size of the points shown in the fixed image.",
        in_app=False,
    )
    size_moving: int = Field(
        10,
        ge=1,
        le=40,
        step_size=1,
        title="Size (moving)",
        description="Size of the points shown in the moving image.",
        in_app=False,
    )
    label_size: int = Field(
        12,
        ge=4,
        le=60,
        step_size=4,
        title="Label size",
        description="Size of the text associated with each label.",
        in_app=False,
    )
    label_color: str = Field(
        "#FFFF00", title="Label color", description="Color of the text associated with each label.", in_app=False
    )
    viewer_orientation: ViewerOrientation = Field(
        ViewerOrientation.VERTICAL, title="Viewer orientation", description="Orientation of the viewer.", in_app=False
    )
    fixed_dir: str = Field("", title="Fixed directory", description="Directory with fixed images.", in_app=False)
    moving_dir: str = Field("", title="Moving directory", description="Directory with moving images.", in_app=False)
    first_time_register: bool = Field(
        True, title="First time", description="First time running the register app.", in_app=True
    )
    confirm_close_register: bool = Field(
        True, title="Confirm close", description="Confirm close register app.", in_app=True
    )

    # Export-app parameters
    first_time_fusion: bool = Field(
        True, title="First time", description="First time running the export app.", in_app=True
    )
    confirm_close_fusion: bool = Field(
        True, title="Confirm close", description="Confirm close export app.", in_app=True
    )

    # Viewer-app parameters
    first_time_viewer: bool = Field(
        True, title="First time", description="First time running the viewer app.", in_app=True
    )
    confirm_close_viewer: bool = Field(
        True, title="Confirm close", description="Confirm close viewer app.", in_app=True
    )

    # WsiPrep-app parameters
    rotate_step_size: int = Field(15, title="Rotate by", description="Rotate by.", in_app=True)
    translate_step_size: int = Field(250, title="Translate by", description="Translate by.", in_app=True)
    view_mode: str = Field("group", title="View mode", description="View mode.", in_app=True)
    project_mode: str = Field("2D (one reference per group)", description="Project mode.", in_app=True)
    slide_tag: str = Field("slide", title="Slide tag", description="Slide tag.", in_app=True)
    project_prefix_tag: str = Field("", title="Project prefix")
    project_suffix_tag: str = Field("", title="Project suffix")
    project_tag: str = Field("group", title="Project group")
    pyramid_level: int = Field(-1, ge=-3, le=-1)
    transformations: tuple[str] = Field(
        ("rigid", "affine"), title="Transformations", description="Transformations.", in_app=True
    )
    common_intensity: bool = Field(True, title="Common intensity for all images")

    first_time_wsiprep: bool = Field(
        True, title="First time", description="First time running the viewer app.", in_app=True
    )
    confirm_close_wsiprep: bool = Field(
        True, title="Confirm close", description="Confirm close viewer app.", in_app=True
    )

    # Viewer-app parameters
    first_time_convert: bool = Field(
        True, title="First time", description="First time running the convert app.", in_app=True
    )
    confirm_close_convert: bool = Field(
        True, title="Confirm close", description="Confirm close convert app.", in_app=True
    )
    as_uint8: bool = Field(True, title="Convert to uint8", description="Convert to uint8.", in_app=True)

    # telemetry
    telemetry_enabled: bool = Field(True, title="Enable telemetry", description="Enable telemetry.", in_app=True)
    telemetry_with_locals: bool = Field(
        True, title="Send locals", description="Send locals with telemetry.", in_app=True
    )

    @validator("fixed_dir", "moving_dir", "output_dir", pre=True, allow_reuse=True)
    def _validate_path(value: PathLike) -> str:  # type: ignore[misc]
        """Validate path."""
        return str(value)

    @validator("viewer_orientation", pre=True, allow_reuse=True)
    def _validate_orientation(value: ty.Union[str, ViewerOrientation]) -> ViewerOrientation:  # type: ignore[misc]
        """Validate path."""
        return ViewerOrientation(value)


class State:
    """State of the application."""

    @property
    def allow_filters(self) -> bool:
        """Allow filters."""
        return True
        # return is_envvar("IMAGE2IMAGE_NO_FILTER", "0") or (not IS_PYINSTALLER and not IS_MAC)


CONFIG: Config = Config()  # type: ignore[call-arg]
STATE = State()
