"""Constant values."""

from image2image.config import STATE

UINT8_TOOLTIP = (
    "Convert to uint8 to reduce file size with minimal data loss. This will result in change of the"
    " dynamic range of the image to between 0-255."
)
UINT8_WARNING = (
    "While this option reduces the amount of space an image takes on your disk, it can lead to data loss<br>"
    " and should be used with caution. If you are exporting RGB images (e.g. H&E or PAS), then this is not<br>"
    " a concert since that data is already in the 0-255 range. However, if you are exporting images with a<br>"
    " dynamic range greater than 0-255, then you should be careful (e.g. AF, IHC, CODEX)."
)

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

# to add apps: volume viewer, sync viewer,
REGISTER_TEXT = "Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "Overlay your microscopy and imaging mass spectrometry data."
CROP_TEXT = "Crop your microscopy data to reduce it's size (handy for Image Fusion)."
CONVERT_TEXT = "Convert multi-scene CZI images or other formats to OME-TIFF."
MERGE_TEXT = "Merge multiple OME-TIFF images into a single file."
FUSION_TEXT = "Export your data for Image Fusion in MATLAB compatible format."
ELASTIX_TEXT = "Register whole slide microscopy images<br>(<b>Elastix</b>)."
VALIS_TEXT = "Register whole slide microscopy images<br>(<b>Valis</b>)."
CONVERT_WARNING = ""
if not STATE.allow_convert:
    CONVERT_WARNING = "<i>Not available on Apple Silicon due to a bug I can't find...</i>"
VALIS_WARNING = ""
if not STATE.allow_valis:
    VALIS_WARNING = "<br><br><i>Might not work without a proper setup.</i>"
