"""Assets."""

from pathlib import Path

from koyo.system import IS_MAC, get_module_path
from qtextra.assets import update_icon_mapping, update_styles
from qtextra.config.theme import THEMES, Theme

HERE = Path(get_module_path("image2image.assets", "__init__.py")).parent.resolve()

ICON_SVG = str(HERE / "icon.svg")
ICON_ICO = str(HERE / ("icon.icns" if IS_MAC else "icon.ico"))
ICON_PNG = str(HERE / "icon.png")


STYLES_PATH = HERE / "stylesheets"
STYLES_PATH.mkdir(exist_ok=True)
update_styles({x.stem: str(x) for x in STYLES_PATH.iterdir() if x.suffix == ".qss"})


update_icon_mapping(
    {
        # other
        "transform": "ri.drag-move-line",
        "preprocess": "fa5s.paint-brush",
        "common": "mdi.form-select",
        "extract": "fa6s.download",
        # wsireg
        "mask": "mdi6.drama-masks",
        "process": "msc.wand",
        # "preview": "mdi6.image-edit",
        "preview": "mdi6.camera-iris",
        "attachment": "ri.attachment-2",
        "shapes": "fa5s.shapes",
        "image": "mdi6.image",
        "geojson": "msc.json",
        "points": "mdi6.scatter-plot",
        # other
        "remove_single": "mdi.close-circle",
        "remove_multiple": "mdi.close-circle-multiple",
        "remove_all": "fa5s.trash-alt",
        "bring_to_top": "fa5s.angle-double-up",
        "fiducial": "fa5s.map-marker-alt",
        "change": "fa5s.arrow-alt-circle-right",
        "swap": "mdi6.swap-vertical-bold",
        "keep_image": "mdi6.check-circle-outline",
        "remove_image": "mdi6.close-circle-outline",
        "env": "fa5s.bookmark",
        "iterate": "mdi.view-carousel-outline",
        # register app
        "fixed": "fa5s.anchor",
        "moving": "ri.drag-move-2-line",
        # app icons
        "viewer": "fa5.eye",
        "elastix": "fa5.images",
        "register": "fa5s.layer-group",
        "convert": "fa5s.arrow-alt-circle-right",
        "sync": "fa5s.sync-alt",
        "crop": "fa5s.crop-alt",
        "fusion": "mdi6.content-save-check-outline",
        "merge": "msc.merge",
        "valis": "fa5b.vimeo-square",
        "launch": "mdi.rocket-launch",
    }
)

THEME = {
    "name": "image2image",
    "type": "light",
    "background": "rgb(247, 247, 247)",
    "foreground": "rgb(245, 224, 218)",
    "primary": "rgb(156, 151, 148)",
    "secondary": "rgb(134, 130, 135)",
    "highlight": "rgb(198, 207, 126)",
    "text": "rgb(13, 13, 13)",
    "icon": "rgb(81, 86, 105)",
    "warning": "rgb(255, 105, 60)",
    "error": "rgb(255, 18, 31)",
    "success": "rgb(12, 237, 91)",
    "progress": "rgb(255, 175, 77)",
    "current": "rgb(12, 237, 91)",
    "syntax_style": "default",
    "console": "rgb(255, 255, 255)",
    "canvas": "rgb(255, 255, 255)",
    "standout": "rgb(255, 252, 0)",
    "font_size": "14px",
    "header_size": "18px",
}

theme = Theme(**THEME)
THEMES.add_theme(theme.name, theme, register=True)
THEMES.register_themes()
# THEMES.theme = "image2image"
THEMES[THEMES.theme].font_size = "9pt"
