"""Configuration."""

import typing as ty

from koyo.config import BaseConfig
from koyo.typing import PathLike
from pydantic import Field, field_serializer, field_validator

from image2image.enums import ViewerOrientation
from image2image.utils._appdirs import USER_CONFIG_DIR


class State:
    """State of the application."""

    _allow_valis_run: ty.Optional[bool] = None

    @property
    def is_mac_arm_pyinstaller(self) -> bool:
        """Check if running on Mac ARM with PyInstaller."""
        from koyo.system import IS_MAC_ARM, IS_PYINSTALLER

        return IS_PYINSTALLER and IS_MAC_ARM

    @property
    def allow_filters(self) -> bool:
        """Allow filters."""
        return True
        # return is_envvar("IMAGE2IMAGE_NO_FILTER", "0") or (not IS_PYINSTALLER and not IS_MAC)

    @property
    def allow_convert(self) -> bool:
        """Allow Convert app."""
        return True
        # from koyo.system import IS_MAC_ARM, IS_PYINSTALLER

        # return IS_PYINSTALLER and IS_MAC_ARM

    @property
    def allow_valis(self) -> bool:
        """Allow Valis app."""
        from koyo.utilities import is_installed

        return is_installed("valis") and is_installed("pyvips")

    @property
    def allow_valis_run(self) -> bool:
        """Allow execution of valis command."""
        import os
        import subprocess

        if not self.allow_valis:
            return False
        if self._allow_valis_run is not None:
            return True
        i2reg_path = os.environ.get("IMAGE2IMAGE_I2REG_PATH", None)
        if not i2reg_path:
            return False
        try:
            ret = subprocess.check_call([i2reg_path, "valis", "--help"])
            self._allow_valis_run = ret == 0
        except subprocess.CalledProcessError:
            self._allow_valis_run = False
        return self._allow_valis_run


class Config(BaseConfig):
    """Configuration of few parameters."""

    USER_CONFIG_DIR = USER_CONFIG_DIR

    # visuals
    theme: str = Field(
        "light",
        title="Theme",
        description="Theme of the application.",
        json_schema_extra={
            "options": ["light", "dark"],
            "in_app": True,
        },
    )

    # telemetry
    telemetry_enabled: bool = Field(
        True,
        title="Enable telemetry",
        description="Enable telemetry.",
        json_schema_extra={
            "in_app": True,
        },
    )
    telemetry_with_locals: bool = Field(
        True,
        title="Send locals",
        description="Send locals with telemetry.",
        json_schema_extra={
            "in_app": True,
        },
    )


class SingleAppConfig(BaseConfig):
    """Basic app configuration"""

    USER_CONFIG_DIR = USER_CONFIG_DIR

    # paths
    output_dir: str = Field(
        "",
        title="Output directory",
        description="Directory where output should be saved.",
        json_schema_extra={
            "in_app": False,
        },
    )
    last_dir: str = Field(
        "",
        title="Last directory",
        description="Last directory used.",
        json_schema_extra={
            "in_app": False,
        },
    )

    # app
    first_time: bool = Field(
        True,
        title="First time",
        description="First time running the viewer app.",
        json_schema_extra={
            "in_app": True,
        },
    )
    confirm_close: bool = Field(
        True,
        title="Confirm close",
        description="Confirm close viewer app.",
        json_schema_extra={
            "in_app": True,
        },
    )

    # export
    as_uint8: bool = Field(
        True,
        title="Convert to uint8",
        description="Convert to uint8.",
        json_schema_extra={
            "in_app": True,
        },
    )
    overwrite: bool = Field(
        False,
        title="Overwrite",
        description="Overwrite.",
        json_schema_extra={
            "in_app": True,
        },
    )
    tile_size: int = Field(
        512,
        title="Tile size",
        description="Tile size.",
        json_schema_extra={
            "in_app": True,
        },
    )

    @field_validator("output_dir", "last_dir", mode="before")
    @classmethod
    def _validate_path(cls, value: PathLike) -> str:  # type: ignore[misc]
        """Validate path."""
        return str(value)

    @field_serializer("output_dir", "last_dir", when_used="always")
    def _serialize_path(self, value: PathLike) -> str:
        """Serialize transformations."""
        return str(value)

    @field_validator("tile_size", mode="before")
    @classmethod
    def _validate_tile_size(cls, value: ty.Union[int, str]) -> int:  # type: ignore[misc]
        """Validate path."""
        return int(value)


class ViewerConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.viewer.json"


class CropConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.crop.json"


class MergeConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.merge.json"


class FusionConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.fusion.json"


class ElastixConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.elastix.json"

    env_i2reg: str = Field(
        "",
        title="i2reg environment variable",
        description="Path to the environment variable that leads to i2reg.",
        json_schema_extra={
            "in_app": True,
        },
    )
    n_parallel: int = Field(
        1,
        ge=1,
        le=8,
        title="Number of parallel processes",
        description="Number of parallel processes.",
        json_schema_extra={
            "in_app": True,
        },
    )

    transformations: tuple[str, ...] = Field(
        ("rigid", "affine"),
        title="Transformations",
        description="Transformations.",
        json_schema_extra={
            "in_app": True,
        },
    )
    use_preview: bool = Field(
        True,
        title="Use preview",
        description="Use preview.",
        json_schema_extra={
            "in_app": True,
        },
    )
    hide_others: bool = Field(
        False,
        title="Hide others",
        description="Hide others.",
        json_schema_extra={
            "in_app": True,
        },
    )
    auto_show: bool = Field(
        True,
        title="Auto show",
        description="Auto show.",
        json_schema_extra={
            "in_app": True,
        },
    )
    open_when_finished: bool = Field(
        True,
        title="Open when finished",
        description="Open when finished.",
        json_schema_extra={
            "in_app": True,
        },
    )

    # writing options
    write_registered: bool = Field(
        True,
        title="Write registered",
        description="Write registered.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_not_registered: bool = Field(
        True,
        title="Write not registered",
        description="Write not registered.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_attached: bool = Field(
        True,
        title="Write attached",
        description="Write attached.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_merged: bool = Field(
        True,
        title="Write merged",
        description="Write merged.",
        json_schema_extra={
            "in_app": True,
        },
    )
    remove_merged: bool = Field(
        False,
        title="Remove merged",
        description="Remove merged.",
        json_schema_extra={
            "in_app": True,
        },
    )
    rename: bool = Field(
        False,
        title="Rename",
        description="Rename.",
        json_schema_extra={
            "in_app": True,
        },
    )
    clip: ty.Literal["ignore", "clip", "remove", "part-remove"] = Field(
        "remove", title="Clip", description="Clip.", json_schema_extra={"in_app": True}
    )

    @field_validator("output_dir", "last_dir", "env_i2reg", mode="before")
    @classmethod
    def _validate_path(cls, value: PathLike) -> str:  # type: ignore[misc]
        """Validate path."""
        return str(value)

    @field_serializer("output_dir", "last_dir", "env_i2reg", when_used="always")
    def _serialize_path(self, value: PathLike) -> str:
        """Serialize transformations."""
        return str(value)

    @field_validator("transformations", mode="before")
    @classmethod
    def _validate_transformations(cls, value: tuple[str, ...]) -> tuple[str, ...]:  # type: ignore[misc]
        """Validate path."""
        if value is None:
            return ()
        return tuple(value)

    @field_serializer("transformations", when_used="always")
    def _serialize_transformations(self, value: tuple[str, ...]) -> tuple[str, ...]:
        """Serialize transformations."""
        return tuple(value)


class ValisConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.valis.json"

    env_i2reg: str = Field(
        "",
        title="i2reg environment variable",
        description="Path to the environment variable that leads to i2reg.",
        json_schema_extra={
            "in_app": True,
        },
    )
    n_parallel: int = Field(
        1,
        ge=1,
        le=8,
        title="Number of parallel processes",
        description="Number of parallel processes.",
        json_schema_extra={
            "in_app": True,
        },
    )

    use_preview: bool = Field(
        True,
        title="Use preview",
        description="Use preview.",
        json_schema_extra={
            "in_app": True,
        },
    )
    hide_others: bool = Field(
        False,
        title="Hide others",
        description="Hide others.",
        json_schema_extra={
            "in_app": True,
        },
    )
    auto_show: bool = Field(
        True,
        title="Auto show",
        description="Auto show.",
        json_schema_extra={
            "in_app": True,
        },
    )
    open_when_finished: bool = Field(
        True,
        title="Open when finished",
        description="Open when finished.",
        json_schema_extra={
            "in_app": True,
        },
    )

    # Valis options
    feature_detector: str = Field(
        "vsgg",
        title="Feature detector",
        description="Feature detector.",
        json_schema_extra={
            "in_app": True,
        },
    )
    feature_matcher: str = Field(
        "ransac",
        title="Feature matcher",
        description="Feature matcher.",
        json_schema_extra={
            "in_app": True,
        },
    )
    check_reflection: bool = Field(
        True,
        title="Check reflections",
        description="Check reflections.",
        json_schema_extra={
            "in_app": True,
        },
    )
    allow_non_rigid: bool = Field(
        False,
        title="Allow non-rigid",
        description="Allow non-rigid.",
        json_schema_extra={
            "in_app": True,
        },
    )
    allow_micro: bool = Field(
        False,
        title="Allow micro",
        description="Allow micro.",
        json_schema_extra={
            "in_app": True,
        },
    )
    micro_fraction: float = Field(
        0.1,
        title="Micro fraction",
        description="Micro fraction.",
        json_schema_extra={
            "in_app": True,
        },
    )

    # writing options
    write_registered: bool = Field(
        True,
        title="Write registered",
        description="Write registered.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_not_registered: bool = Field(
        True,
        title="Write not registered",
        description="Write not registered.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_attached: bool = Field(
        True,
        title="Write attached",
        description="Write attached.",
        json_schema_extra={
            "in_app": True,
        },
    )
    write_merged: bool = Field(
        True,
        title="Write merged",
        description="Write merged.",
        json_schema_extra={
            "in_app": True,
        },
    )
    remove_merged: bool = Field(
        False,
        title="Remove merged",
        description="Remove merged.",
        json_schema_extra={
            "in_app": True,
        },
    )
    rename: bool = Field(
        False,
        title="Rename",
        description="Rename.",
        json_schema_extra={
            "in_app": True,
        },
    )
    clip: ty.Literal["ignore", "clip", "remove", "part-remove"] = Field(
        "remove", title="Clip", description="Clip.", json_schema_extra={"in_app": True}
    )

    @field_validator("output_dir", "last_dir", "env_i2reg", mode="before")
    @classmethod
    def _validate_path(cls, value: PathLike) -> str:  # type: ignore[misc]
        """Validate path."""
        return str(value)

    @field_serializer("output_dir", "last_dir", "env_i2reg", when_used="always")
    def _serialize_path(self, value: PathLike) -> str:
        """Serialize transformations."""
        return str(value)


class ConvertConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.convert.json"


class Elastix3dConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.elastix3d.json"

    rotate_step_size: int = Field(
        15,
        title="Rotate by",
        description="Rotate by.",
        json_schema_extra={
            "in_app": True,
        },
    )
    translate_step_size: int = Field(
        250,
        title="Translate by",
        description="Translate by.",
        json_schema_extra={
            "in_app": True,
        },
    )
    view_mode: str = Field(
        "group",
        title="View mode",
        description="View mode.",
        json_schema_extra={
            "in_app": True,
        },
    )
    project_mode: str = Field(
        "2D (one reference per group)",
        description="Project mode.",
        json_schema_extra={
            "in_app": True,
        },
    )
    slide_tag: str = Field(
        "slide",
        title="Slide tag",
        description="Slide tag.",
        json_schema_extra={
            "in_app": True,
        },
    )
    project_prefix_tag: str = Field("", title="Project prefix")
    project_suffix_tag: str = Field("", title="Project suffix")
    project_tag: str = Field("group", title="Project group")
    pyramid_level: int = Field(-1, ge=-3, le=-1)
    transformations: tuple[str, ...] = Field(
        ("rigid", "affine"),
        title="Transformations",
        description="Transformations.",
        json_schema_extra={
            "in_app": True,
        },
    )
    common_intensity: bool = Field(True, title="Common intensity for all images")

    @field_validator("transformations", mode="before")
    @classmethod
    def _validate_transformations(cls, value: ty.Tuple[str, ...]) -> ty.Tuple[str, ...]:  # type: ignore[misc]
        """Validate path."""
        if value is None:
            return ()
        return tuple(value)

    @field_serializer("transformations", when_used="always")
    def _serialize_transformations(self, value: tuple[str, ...]) -> tuple[str, ...]:
        """Serialize transformations."""
        return tuple(value)


class RegisterConfig(SingleAppConfig):
    """Configuration for the register app."""

    USER_CONFIG_FILENAME = "config.register.json"

    enable_prediction: bool = Field(
        True,
        title="Enable prediction",
        description="Enable prediction.",
        json_schema_extra={
            "in_app": True,
        },
    )
    sync_views: bool = Field(
        True,
        title="Sync views",
        description="Sync views.",
        json_schema_extra={
            "in_app": False,
        },
    )
    zoom_factor: float = Field(
        7.5,
        ge=0.5,
        le=100.0,
        title="Zoom factor",
        description="Zoom factor.",
        json_schema_extra={
            "step_size": 0.25,
            "n_decimals": 2,
        },
    )
    opacity_fixed: int = Field(
        100,
        ge=0,
        le=100,
        title="Opacity (fixed)",
        description="Opacity of the fixed image",
        json_schema_extra={
            "in_app": False,
            "step_size": 10,
        },
    )
    opacity_moving: int = Field(
        75,
        ge=0,
        le=100,
        title="Opacity (moving)",
        description="Opacity of the moving image",
        json_schema_extra={
            "in_app": False,
            "step_size": 10,
        },
    )
    size_fixed: int = Field(
        3,
        ge=1,
        le=40,
        title="Size (fixed)",
        description="Size of the points shown in the fixed image.",
        json_schema_extra={
            "in_app": False,
            "step_size": 1,
        },
    )
    size_moving: int = Field(
        10,
        ge=1,
        le=40,
        title="Size (moving)",
        description="Size of the points shown in the moving image.",
        json_schema_extra={
            "in_app": False,
            "step_size": 1,
        },
    )
    label_size: int = Field(
        12,
        ge=4,
        le=60,
        title="Label size",
        description="Size of the text associated with each label.",
        json_schema_extra={
            "in_app": False,
            "step_size": 4,
        },
    )
    label_color: str = Field(
        "#FFFF00",
        title="Label color",
        description="Color of the text associated with each label.",
        json_schema_extra={
            "in_app": False,
        },
    )
    viewer_orientation: ViewerOrientation = Field(
        ViewerOrientation.VERTICAL,
        title="Viewer orientation",
        description="Orientation of the viewer.",
        json_schema_extra={
            "in_app": False,
        },
    )
    fixed_dir: str = Field(
        "",
        title="Fixed directory",
        description="Directory with fixed images.",
        json_schema_extra={
            "in_app": False,
        },
    )
    moving_dir: str = Field(
        "",
        title="Moving directory",
        description="Directory with moving images.",
        json_schema_extra={
            "in_app": False,
        },
    )

    # fiducial marker options
    zoom_on_point: bool = Field(
        True,
        title="Fiducial zoom",
        description="Fiducial zoom.",
        json_schema_extra={
            "in_app": False,
        },
    )
    simplify_contours_distance: float = Field(
        2.25,
        title="Simplify contour",
        description="Simplify contour.",
        le=10,
        ge=0,
        json_schema_extra={
            "in_app": False,
        },
    )
    zoom_scale: float = Field(
        0.01,
        title="Zoom scale",
        description="Zoom scale.",
        ge=0.0001,
        le=1,
        json_schema_extra={
            "in_app": False,
        },
    )

    @field_validator("viewer_orientation", mode="before")
    @classmethod
    def _validate_orientation(cls, value: ty.Union[str, ViewerOrientation]) -> ViewerOrientation:  # type: ignore[misc]
        """Validate path."""
        return ViewerOrientation(value)


STATE = State()

APP_CONFIG: ty.Optional[Config] = None


def get_app_config() -> Config:
    """Get App config."""

    global APP_CONFIG
    if APP_CONFIG is None:
        APP_CONFIG = Config(_auto_load=True)  # type: ignore[call-arg]
    return APP_CONFIG


VIEWER_CONFIG: ty.Optional[ViewerConfig] = None


def get_viewer_config() -> ViewerConfig:
    """Get Viewer config."""

    global VIEWER_CONFIG
    if VIEWER_CONFIG is None:
        VIEWER_CONFIG = ViewerConfig(_auto_load=True)  # type: ignore[call-arg]
    return VIEWER_CONFIG


CROP_CONFIG: ty.Optional[CropConfig] = None


def get_crop_config() -> CropConfig:
    """Get Crop config."""

    global CROP_CONFIG
    if CROP_CONFIG is None:
        CROP_CONFIG = CropConfig(_auto_load=True)  # type: ignore[call-arg]
    return CROP_CONFIG


MERGE_CONFIG: ty.Optional[MergeConfig] = None


def get_merge_config() -> MergeConfig:
    """Get Merge config."""

    global MERGE_CONFIG
    if MERGE_CONFIG is None:
        MERGE_CONFIG = MergeConfig(_auto_load=True)  # type: ignore[call-arg]
    return MERGE_CONFIG


FUSION_CONFIG: ty.Optional[FusionConfig] = None


def get_fusion_config() -> FusionConfig:
    """Get Fusion config."""

    global FUSION_CONFIG
    if FUSION_CONFIG is None:
        FUSION_CONFIG = FusionConfig(_auto_load=True)  # type: ignore[call-arg]
    return FUSION_CONFIG


CONVERT_CONFIG: ty.Optional[ConvertConfig] = None


def get_convert_config() -> ConvertConfig:
    """Get Convert config."""

    global CONVERT_CONFIG
    if CONVERT_CONFIG is None:
        CONVERT_CONFIG = ConvertConfig(_auto_load=True)  # type: ignore[call-arg]
    return CONVERT_CONFIG


REGISTER_CONFIG: ty.Optional[RegisterConfig] = None


def get_register_config() -> RegisterConfig:
    """Get Register config."""

    global REGISTER_CONFIG
    if REGISTER_CONFIG is None:
        REGISTER_CONFIG = RegisterConfig(_auto_load=True)  # type: ignore[call-arg]
    return REGISTER_CONFIG


VALIS_CONFIG: ty.Optional[ValisConfig] = None


def get_valis_config() -> ValisConfig:
    """Get Valis config."""

    global VALIS_CONFIG
    if VALIS_CONFIG is None:
        VALIS_CONFIG = ValisConfig(_auto_load=True)  # type: ignore[call-arg]
    return VALIS_CONFIG


ELASTIX_CONFIG: ty.Optional[ElastixConfig] = None


def get_elastix_config() -> ElastixConfig:
    """Get Elastix config."""

    global ELASTIX_CONFIG
    if ELASTIX_CONFIG is None:
        ELASTIX_CONFIG = ElastixConfig(_auto_load=True)  # type: ignore[call-arg]
    return ELASTIX_CONFIG


ELASTIX3D_CONFIG: ty.Optional[Elastix3dConfig] = None


def get_elastix3d_config() -> Elastix3dConfig:
    """Get Elastix3d config."""

    global ELASTIX3D_CONFIG
    if ELASTIX3D_CONFIG is None:
        ELASTIX3D_CONFIG = Elastix3dConfig(_auto_load=True)  # type: ignore[call-arg]
    return ELASTIX3D_CONFIG
