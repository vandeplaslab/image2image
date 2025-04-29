"""Enums."""

from enum import auto

from image2image_io.enums import ViewType
from napari.utils.misc import StringEnum

ALLOWED_PROJECT_IMPORT_REGISTER_FORMATS = (
    "Any transformation (*.i2r.json *.i2r.toml *.toml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "imsmicrolink transformation (*.toml);;"
)
ALLOWED_PROJECT_EXPORT_REGISTER_FORMATS = (
    "Any transformation (*.json *.toml *.xml);; "
    "Transformation (*.i2r.json);; "
    "Transformation (*.i2r.toml);;"
    "MATLAB fusion format (*.xml);;"
)
ALLOWED_PROJECT_VIEWER_FORMATS = (
    "Any projects (*.json *.i2v.json *.i2v.toml);; JSON project (*.i2v.json);; TOML project (*.i2v.toml);;"
)
ALLOWED_PROJECT_CROP_FORMATS = (
    "Any projects (*.i2c.json *.i2c.toml);; JSON project (*.i2c.json);; TOML project (*.i2c.toml);;"
)
ALLOWED_PROJECT_WSIPREP_FORMATS = (
    "Any projects (*.i2wsiprep.json *.i2wsiprep.toml);; "
    "JSON project (*.i2wsiprep.json);; "
    "TOML project (*.i2wsiprep.toml);;"
)
ALLOWED_PROJECT_ELASTIX_FORMATS = (
    "Any projects (*.json *.toml *.i2wsireg.json *.i2wsireg.toml *.wsireg *.i2reg *.config.json);; "
    "JSON project (*.json *.i2reg.json);; "
    "TOML project (*.toml *.i2reg.toml);;"
    "I2Reg project (*.wsireg *.i2reg *.config.json);;"
)

ALLOWED_PROJECT_VALIS_FORMATS = (
    "Any projects (*.json *.valis.json *.valis.toml *.valis *.config.json);; "
    "JSON project (*.json *.valis.json);; "
    "TOML project (*.toml *.valis.toml);;"
    "Valis project (*.valis *.config.json);;"
)

ALLOWED_IMAGE_FORMATS = (
    "Any imaging (*.d *.tsf *.tdf *.imzML  *.ibd *.data *.metadata.h5 peaks_*.h5 *.npy *.npz *.czi *.ome.tiff *.tiff"
    "*.scn *.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi *.jpg *.jpeg *.png);; "
    "Bruker (*.d *.tsf *.tdf);; "
    "imzML (*.imzML *.ibd);; "
    "ionglow (*.data *.metadata.h5 peaks_*.h5);;"
    "Numpy (*.npy *.npz);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
)
ALLOWED_IMAGE_FORMATS_WITH_GEOJSON = (
    "Any imaging (*.d *.tsf *.tdf *.imzML  *.ibd *.data *.metadata.h5 peaks_*.h5 *.npy *.npz *.czi *.ome.tiff *.tiff"
    "*.scn*.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi *.jpg *.jpeg *.png *.geojson *.json *.csv"
    "*.tsv *.txt *.parquet);; "
    "Bruker (*.d *.tsf *.tdf);; "
    "imzML (*.imzML *.ibd);; "
    "ionglow (*.data *.metadata.h5 peaks_*.h5);;"
    "Numpy (*.npy *.npz);;"
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
    "GeoJSON (*.geojson *.json);;"
    "Points (*.txt *.tsv *.csv *.parquet);;"
)
ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY = (
    "Any microscopy (*.czi *.ome.tiff *.tiff *.scn"
    "*.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi *.jpg"
    " *.jpeg *.png *.geojson *.json);; "
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
    "JPEG (*.jpg *.jpeg);; "
    "PNG (*.png);;"
)
ALLOWED_IMAGE_FORMATS_CZI_ONLY = "Any Zeiss CZI (*.czi);;"
ALLOWED_IMAGE_FORMATS_TIFF_ONLY = (
    "Any OME-TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
)
ALLOWED_ELASTIX_FORMATS = (
    "Any imaging (*.czi *.ome.tiff *.tiff *.scn *.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi);; "
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
)
ALLOWED_VALIS_FORMATS = (
    "Any imaging (*.czi *.ome.tiff *.tiff *.scn *.tif *.qptiff *.qptiff.raw *.qptiff.intermediate *.svs *.ndpi);; "
    "CZI (*.czi);; "
    "TIFF (*.ome.tiff *.tiff *.scn *.tif *.svs *.ndpi *.qptiff *.qptiff.raw *.qptiff.intermediate);; "
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


VIEW_TYPE_TRANSLATIONS = {
    ViewType.RANDOM: "Random",
    ViewType.OVERLAY: "Overlay",
}

REGISTRATION_PATH_HELP = (
    "Type or registration path. There are several types including:<br>"
    "<b>rigid</b> - Translation and rotation<br>"
    "<b>similarity</b> - Translation, rotation, and scale<br>"
    "<b>affine</b> - Translation, rotation, scale, and shear<br>"
    "<b>nl</b> - Non-linear registration<br>"
    "<br>The suffix of the registration also indicates the type:<br>"
    "<b>reduced</b> - Quicker registration with fewer parameters or optimization steps<br>"
    "<b>mid</b> - Medium registration with more parameters or optimization steps<br>"
    "<b>expanded</b> - Slower registration with more optimization steps<br>"
    "<b>extreme</b> - Even slower registration with more optimization steps<br>"
    "<br>Additionally, these suffixes will indicate which metric is used during the registration<br>"
    "<b>ams</b> - Advanced Mean Squares metric instead of Advanced Mattes Mutual Information<br>"
    "<b>anc</b> - Advanced Normalized Correlation metric instead of Advanced Mattes Mutual Information<br>"
)


PYRAMID_TO_LEVEL = {
    "-1 (smallest pyramid level)": -1,
    "-2 (second smallest pyramid level)": -2,
    "-3 (third smallest pyramid level)": -3,
    "0 (full resolution image - not recommended)": 0,
}
LEVEL_TO_PYRAMID = {v: k for k, v in PYRAMID_TO_LEVEL.items()}
