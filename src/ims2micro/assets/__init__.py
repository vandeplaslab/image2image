"""Assets."""
from pathlib import Path
from qtextra.assets import update_icon_mapping, update_styles, update_icons
from qtextra.utils.utilities import get_module_path
from qtextra.config import THEMES


HERE = Path(get_module_path("ims2micro.assets", "__init__.py")).parent.resolve()

icon_path = HERE / "icons"
icon_path.mkdir(exist_ok=True)
update_icons({x.stem: str(x) for x in icon_path.iterdir() if x.suffix == ".svg"})


styles_path = HERE / "stylesheets"
styles_path.mkdir(exist_ok=True)
update_styles({x.stem: str(x) for x in styles_path.iterdir() if x.suffix == ".qss"})


update_icon_mapping(
    {
        # "layers": "fa5s.layer-group",
        # "cross_full": "fa5s.times-circle",
        "remove": "ri.indeterminate-circle-line",
        "add": "ri.add-circle-fill",
        "zoom": "mdi.magnify",
        # "open": "fa5s.folder-open",
        "close": "fa5s.trash-alt",
        # "folder": "mdi.folder-move-outline",
    }
)
THEMES.register_themes()
