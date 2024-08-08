"""Configuration."""

import typing as ty

from koyo.config import BaseConfig
from koyo.typing import PathLike
from pydantic import Field, validator

from image2image.enums import ViewerOrientation
from image2image.utils._appdirs import USER_CONFIG_DIR


class State:
    """State of the application."""

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


class Config(BaseConfig):
    """Configuration of few parameters."""

    USER_CONFIG_DIR = USER_CONFIG_DIR

    # visuals
    theme: str = Field(
        "light", title="Theme", description="Theme of the application.", options=["light", "dark"], in_app=True
    )

    # telemetry
    telemetry_enabled: bool = Field(True, title="Enable telemetry", description="Enable telemetry.", in_app=True)
    telemetry_with_locals: bool = Field(
        True, title="Send locals", description="Send locals with telemetry.", in_app=True
    )


class SingleAppConfig(BaseConfig):
    """Basic app configuration"""

    USER_CONFIG_DIR = USER_CONFIG_DIR

    # paths
    output_dir: str = Field(
        "", title="Output directory", description="Directory where output should be saved.", in_app=False
    )
    last_dir: str = Field("", title="Last directory", description="Last directory used.", in_app=False)

    # app
    first_time: bool = Field(True, title="First time", description="First time running the viewer app.", in_app=True)
    confirm_close: bool = Field(True, title="Confirm close", description="Confirm close viewer app.", in_app=True)

    # export
    as_uint8: bool = Field(True, title="Convert to uint8", description="Convert to uint8.", in_app=True)
    overwrite: bool = Field(False, title="Overwrite", description="Overwrite.", in_app=True)
    tile_size: str = Field(512, title="Tile size", description="Tile size.", in_app=True)

    @validator("output_dir", "last_dir", pre=True, allow_reuse=True)
    def _validate_path(value: PathLike) -> str:  # type: ignore[misc]
        """Validate path."""
        return str(value)


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

    transformations: tuple[str] = Field(
        ("rigid", "affine"), title="Transformations", description="Transformations.", in_app=True
    )
    use_preview: bool = Field(True, title="Use preview", description="Use preview.", in_app=True)
    hide_others: bool = Field(False, title="Hide others", description="Hide others.", in_app=True)
    open_when_finished: bool = Field(True, title="Open when finished", description="Open when finished.", in_app=True)

    # writing options
    write_registered: bool = Field(True, title="Write registered", description="Write registered.", in_app=True)
    write_not_registered: bool = Field(
        True, title="Write not registered", description="Write not registered.", in_app=True
    )
    write_attached: bool = Field(True, title="Write attached", description="Write attached.", in_app=True)
    write_merged: bool = Field(True, title="Write merged", description="Write merged.", in_app=True)
    remove_merged: bool = Field(False, title="Remove merged", description="Remove merged.", in_app=True)
    rename: bool = Field(True, title="Rename", description="Rename.", in_app=True)


class ValisConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.valis.json"

    use_preview: bool = Field(True, title="Use preview", description="Use preview.", in_app=True)
    hide_others: bool = Field(False, title="Hide others", description="Hide others.", in_app=True)
    open_when_finished: bool = Field(True, title="Open when finished", description="Open when finished.", in_app=True)

    # Valis options
    feature_detector: str = Field("vsgg", title="Feature detector", description="Feature detector.", in_app=True)
    feature_matcher: str = Field("ransac", title="Feature matcher", description="Feature matcher.", in_app=True)
    check_reflection: bool = Field(True, title="Check reflections", description="Check reflections.", in_app=True)
    allow_non_rigid: bool = Field(False, title="Allow non-rigid", description="Allow non-rigid.", in_app=True)
    allow_micro: bool = Field(False, title="Allow micro", description="Allow micro.", in_app=True)
    micro_fraction: float = Field(0.1, title="Micro fraction", description="Micro fraction.", in_app=True)

    # writing options
    write_registered: bool = Field(True, title="Write registered", description="Write registered.", in_app=True)
    write_not_registered: bool = Field(
        True, title="Write not registered", description="Write not registered.", in_app=True
    )
    write_attached: bool = Field(True, title="Write attached", description="Write attached.", in_app=True)
    write_merged: bool = Field(True, title="Write merged", description="Write merged.", in_app=True)
    remove_merged: bool = Field(False, title="Remove merged", description="Remove merged.", in_app=True)
    rename: bool = Field(True, title="Rename", description="Rename.", in_app=True)


class ConvertConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.convert.json"


class Elastix3dConfig(SingleAppConfig):
    """Configuration for the viewer app."""

    USER_CONFIG_FILENAME = "config.elastix3d.json"

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


class RegisterConfig(SingleAppConfig):
    """Configuration for the register app."""

    USER_CONFIG_FILENAME = "config.register.json"

    sync_views: bool = Field(True, title="Sync views", description="Sync views.", in_app=False)
    zoom_factor: float = Field(
        7.5, ge=0.5, le=100.0, step_size=0.25, n_decimals=2, title="Zoom factor", description="Zoom factor."
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

    @validator("viewer_orientation", pre=True, allow_reuse=True)
    def _validate_orientation(value: ty.Union[str, ViewerOrientation]) -> ViewerOrientation:  # type: ignore[misc]
        """Validate path."""
        return ViewerOrientation(value)


STATE = State()

APP_CONFIG: Config = Config(_auto_load=True)  # type: ignore[call-arg]

VIEWER_CONFIG: ViewerConfig = ViewerConfig(_auto_load=True)  # type: ignore[call-arg]
CROP_CONFIG: CropConfig = CropConfig(_auto_load=True)  # type: ignore[call-arg]
MERGE_CONFIG: MergeConfig = MergeConfig(_auto_load=True)  # type: ignore[call-arg]
FUSION_CONFIG: FusionConfig = FusionConfig(_auto_load=True)  # type: ignore[call-arg]
VALIS_CONFIG: ValisConfig = ValisConfig(_auto_load=True)  # type: ignore[call-arg]
CONVERT_CONFIG: ConvertConfig = ConvertConfig(_auto_load=True)  # type: ignore[call-arg]
ELASTIX_CONFIG: ElastixConfig = ElastixConfig(_auto_load=True)  # type: ignore[call-arg]
ELASTIX3D_CONFIG: Elastix3dConfig = Elastix3dConfig(_auto_load=True)  # type: ignore[call-arg]
REGISTER_CONFIG: RegisterConfig = RegisterConfig(_auto_load=True)  # type: ignore[call-arg]
