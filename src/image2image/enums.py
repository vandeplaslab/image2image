"""Enums."""
from enum import auto

from napari.utils.misc import StringEnum

ALLOWED_IMPORT_REGISTER_FORMATS = (
    "Any transformation (*.i2r.json *.i2r.toml *.toml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "imsmicrolink transformation (*.toml);;"
)
ALLOWED_EXPORT_REGISTER_FORMATS = (
    "Any transformation (*.json *.toml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "MATLAB fusion format (*.xml);;"
)
ALLOWED_VIEWER_FORMATS = (
    "Any projects (*.i2v.json *.i2v.toml);; " "JSON Project (*.i2v.json);; " "TOML Project (*.i2v.toml);;"
)
ALLOWED_CROP_FORMATS = (
    "Any projects (*.i2c.json *.i2c.toml);; " "JSON Project (*.i2c.json);; " "TOML Project (*.i2c.toml);;"
)
ALLOWED_FORMATS = (
    "Any imaging (*.tsf *.tdf *.imzML *.metadata.h5 peaks_*.h5 *.npy *.czi *.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi"
    " *.jpg *.jpeg *.png);; "
    "Bruker (*.tsf *.tdf);; "
    "imzML (*.imzML);; "
    "ionglow (*.metadata.h5, peaks_*.h5);;"
    "Numpy (*.npy);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
)
ALLOWED_FORMATS_WITH_GEOJSON = (
    "Any imaging (*.tsf *.tdf *.imzML *.metadata.h5 peaks_*.h5 *.npy *.czi *.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi"
    " *.jpg *.jpeg *.png *.geojson *.json);; "
    "Bruker (*.tsf *.tdf);; "
    "imzML (*.imzML);; "
    "ionglow (*.metadata.h5, peaks_*.h5);;"
    "Numpy (*.npy);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
    "GeoJSON (*.geojson *.json);;"
)


class ImageTransformation(StringEnum):
    """Image transformation."""

    EUCLIDEAN = auto()
    SIMILARITY = auto()
    PROJECTIVE = auto()
    AFFINE = auto()


TRANSFORMATION_TRANSLATIONS = {
    ImageTransformation.EUCLIDEAN: "Euclidean",
    ImageTransformation.SIMILARITY: "Similarity",
    ImageTransformation.PROJECTIVE: "Projective",
    ImageTransformation.AFFINE: "Affine",
}


class ViewerOrientation(StringEnum):
    """Viewer orientation."""

    VERTICAL = auto()
    HORIZONTAL = auto()


ORIENTATION_TRANSLATIONS = {
    ViewerOrientation.VERTICAL: "Vertical",
    ViewerOrientation.HORIZONTAL: "Horizontal",
}


class ViewType(StringEnum):
    """View type."""

    RANDOM = auto()
    OVERLAY = auto()


VIEW_TYPE_TRANSLATIONS = {
    ViewType.RANDOM: "Random",
    ViewType.OVERLAY: "Overlay",
}

DEFAULT_TRANSFORM_NAME: str = "Identity matrix"

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"
