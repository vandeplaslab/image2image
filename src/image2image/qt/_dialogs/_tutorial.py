"""Tutorial dialog."""

from __future__ import annotations

import typing as ty

if ty.TYPE_CHECKING:
    from qtextra.widgets.qt_tutorial import TutorialStep

    from image2image.qt.dialog_convert import ImageConvertWindow
    from image2image.qt.dialog_crop import ImageCropWindow
    from image2image.qt.dialog_elastix import ImageElastixWindow
    from image2image.qt.dialog_fusion import ImageFusionWindow
    from image2image.qt.dialog_merge import ImageMergeWindow
    from image2image.qt.dialog_register import ImageRegistrationWindow
    from image2image.qt.dialog_valis import ImageValisWindow
    from image2image.qt.dialog_viewer import ImageViewerWindow


OPEN_PROJECT = "If you've previously saved a project, you can open it here."
ADD_IMAGES = (
    "Click here to select images to open in the app. You can also drag-and-drop images into the app and they will be"
    " opened if the file format is supported.."
)
MANAGE_SELECTION = "You can control what images should be loaded, which image channels should be displayed"
MORE_OPTIONS = "You can change select which channels should be shown, update the pixel size or extract ion images here."
CLIPBOARD = (
    "You can take a screenshot of the canvas and copy it to the clipboard by clicking here! If you right-click"
    " on the button, a few extra options will be shown."
)
SCREENSHOT = (
    "You can save screenshot of the canvas by clicking here!. If you right-click on the button, a few extra options"
    " will be shown."
)
SCALE_BAR = "You can show/hide scalebar on the canvas by clicking here. Make sure to set the right pixel size!."
TUTORIAL = "If you wish to see this tutorial again at a future date, you can click here to show it."
FEEDBACK = "If you have some feedback, don't hesitate to send! You can do it directly in the app!"


def show_merge_tutorial(widget: ImageMergeWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to merge2tiff!",
                message="We would like to show you around before you get started!<br>This app allows you to merge"
                " multiple OME-TIFFs into a single OME-TIFF file. The images should be <b>registered</b>, have the "
                "<b>same</b> image shape and pixel spacing."
                "<br><br>Note. Some of the metadata might be lost during the process and we don't support"
                " 3D or 4D+ images at the moment and are unlikely to do so in the future.",
                widget=widget._image_widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="List of images",
                message="Here is a list of images that will be merged to OME-TIFF format. You can edit image name,"
                " channel names and select what should be retained (or removed).",
                widget=widget.table,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Output directory",
                message="Location where the image will be saved to.",
                widget=widget.directory_btn,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Image name",
                message="Name of the saved image. You don't have to put the .ome.tiff as it will be automatically added"
                " when saving.",
                widget=widget.name_edit,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Modify output settings",
                message="You can control certain parameters of the OME-TIFF such as the tile size or if the image"
                " data type should be changed to uint8 which greatly reduces file size.",
                widget=widget.as_uint8,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Merge to OME-TIFF",
                message="Click here to start the merging process.",
                widget=widget.export_btn,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.BOTTOM_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_convert_tutorial(widget: ImageConvertWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2tiff!",
                message="We would like to show you around before you get started!<br>This app allows you to convert"
                " many microscopy images to OME-TIFF format. This can be particularly useful if you have non-tiled"
                " TIFF images that are slow to load or multi-scene CZI image that are not supported in some"
                " applications. This app lets you convert each scene to a OME-TIFF file which can be opened in other"
                " software.<br><br>Note. Some of the metadata might be lost during the process and we don't support"
                " 3D or 4D+ images at the moment and are unlikely to do so in the future.",
                widget=widget._image_widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="List of images",
                message="Here is a list of images that will be converted to OME-TIFF format. If the CZI image has"
                " multiple scenes (no. scenes), each scene will be converted to a separate OME-TIFF file.",
                widget=widget.table,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Output directory",
                message="Location where the image will be saved to.",
                widget=widget.directory_btn,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Convert to OME-TIFF",
                message="Click here to start the conversion process.",
                widget=widget.export_btn,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Modify output settings",
                message="You can control certain parameters of the OME-TIFF such as the tile size or if the image"
                " data type should be changed to uint8 which greatly reduces file size.",
                widget=widget.as_uint8,
                position=Position.BOTTOM,
                func=(widget.export_btn.cancel_btn.hide,),
            ),
            TutorialStep(
                title="Cancel conversion",
                message="You can always stop conversion process by clicking here. The task will be stopped once the"
                " current step (e.g. scene) is finished.",
                widget=widget.export_btn.cancel_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.export_btn.cancel_btn.show,),
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.export_btn.cancel_btn.hide,),
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.BOTTOM_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_fusion_tutorial(widget: ImageFusionWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2fusion!",
                message="We would like to show you around before you get started!<br>This app allows you to convert"
                " OME-TIFF images to CSV format that is compatible with Raf's Fusion toolbox.<br><br>"
                " Note: This app can use crazy amounts of memory in the conversion process (unavoidable sadly). You can"
                " try to use the <b>image2crop</b> app to crop a region of interest which can be used in the Fusion"
                " toolbox.",
                widget=widget._image_widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="List of images",
                message="Here are all the images that will be converted to Fusion CSV format.",
                widget=widget.table,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Output directory",
                message="Location where the image will be saved to.",
                widget=widget.directory_btn,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Convert to CSV",
                message="Click here to start the conversion process.",
                widget=widget.export_btn,
                position=Position.BOTTOM,
                func=(widget.export_btn.cancel_btn.hide,),
            ),
            TutorialStep(
                title="Cancel conversion",
                message="You can always stop conversion process by clicking here. The task should stop almost"
                " immediately.",
                widget=widget.export_btn.cancel_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.export_btn.cancel_btn.show,),
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.export_btn.cancel_btn.hide,),
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.BOTTOM_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def _generic_statusbar(
    widget: ImageViewerWindow | ImageCropWindow | ImageElastixWindow | ImageValisWindow,
) -> list[TutorialStep]:
    from qtextra.widgets.qt_tutorial import Position, TutorialStep

    return [
        TutorialStep(
            title="Save screenshot",
            message=SCREENSHOT,
            widget=widget.screenshot_btn,  # type: ignore[has-type]
            position=Position.BOTTOM_RIGHT,
        ),
        TutorialStep(
            title="Screenshot to clipboard",
            message=CLIPBOARD,
            widget=widget.clipboard_btn,
            position=Position.BOTTOM_RIGHT,
        ),
        TutorialStep(
            title="Show scalebar",
            message=SCALE_BAR,
            widget=widget.scalebar_btn,
            position=Position.BOTTOM_RIGHT,
        ),
        TutorialStep(
            title="Feedback",
            message=FEEDBACK,
            widget=widget.feedback_btn,
            position=Position.BOTTOM_RIGHT,
        ),
        TutorialStep(
            title="Tutorial",
            message=TUTORIAL,
            widget=widget.tutorial_btn,
            position=Position.BOTTOM_RIGHT,
        ),
    ]


def show_register_tutorial(widget: ImageRegistrationWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2register!",
                message="We would like to show you around before you get started!<br>This app let's you generate"
                " image registration information between e.g. microscopy and IMS data. This is done by computing affine"
                " transformation based on fiducial markers in the <b>fixed</b> and <b>moving</b> images.",
                widget=widget.view_fixed.widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="Fixed image canvas",
                message="This is where the <b>fixed</b> (or target) images will be displayed.",
                widget=widget.view_fixed.widget,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Moving image canvas",
                message="This is where the <b>moving</b> images will be displayed.",
                widget=widget.view_moving.widget,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Open previous project",
                message="If you've previously saved a project, you can open it here.",
                widget=widget.import_project_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Control what should be shown",
                message="You can control what images should be loaded and  which image channels should be displayed.",
                widget=widget._fixed_widget,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Add or remove image",
                message=ADD_IMAGES,
                widget=widget._fixed_widget.add_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Control what should be shown",
                message=MORE_OPTIONS,
                widget=widget._fixed_widget.more_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="View type",
                message="You can control how the <b>moving</b> modality should be displayed. The <b>random</b> view is"
                " a good starting point as it ensures that each pixel in an image is visible. The <b>overlay</b>"
                " view will show the ion image.",
                widget=widget._moving_widget.view_type_choice,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Overlay",
                message="You can display <b>one</b> <b>moving</b> image on top of the <b>fixed</b> image. If you've"
                " loaded multiple, you can select which one should be displayed.",
                widget=widget._moving_widget.displayed_in_fixed_choice,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Show fiducials table",
                message="You can show/hide the fiducials table by clicking here. This can be useful when you want to "
                " revisit previous points (<b>double-click</b> on the row to zoom-in) to edit or remove it.",
                widget=widget.fiducials_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Save project",
                message="You can save the current state of the registration (loaded images, fiducial markers,"
                " transformations) and reload it in the future without all that faffing about! The transformation"
                " data can also be used by e.g. <b>AutoIMS</b> or <b>image2viewer</b> app.",
                widget=widget.export_project_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Tutorial",
                message=TUTORIAL,
                widget=widget.tutorial_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Feedback",
                message=FEEDBACK,
                widget=widget.feedback_btn,
                position=Position.BOTTOM_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_viewer_tutorial(widget: ImageViewerWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    def get_transform_widget():
        """Find the transform widget."""
        button = None
        for widget_ in widget._image_widget.dset_dlg._list.widget_iter():
            button = widget_.transform_btn
            if button.isVisible():
                return button
        return button

    transform_btn = get_transform_widget()
    if not transform_btn:
        return None

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2viewer!",
                message="We would like to show you around before you get started!<br>This app let's overlay multiple"
                " images (such as microscopy or IMS), apply appropriate image transformation and export as figures.",
                widget=widget.view.widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="Open previous project",
                message=OPEN_PROJECT,
                widget=widget._image_widget.import_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Add or remove image",
                message=ADD_IMAGES,
                widget=widget._image_widget.add_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Save project",
                message="You can save the current state of the viewer (loaded images, pixel size and transformation"
                " information) and reload it in the future without all that faffing about!",
                widget=widget._image_widget.export_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Control what should be shown",
                message=MORE_OPTIONS,
                widget=widget._image_widget.more_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Transform selection",
                message="When you load certain images (e.g. IMS), you might need to adjust the way it's shown in the "
                "canvas. Usually you've already co-registered your microscopy images but the IMS data won't lie in the"
                " right place as it has different pixel size, spatial orientation or position. You can load previous"
                " transformation data here and apply it to appropriate dataset.",
                widget=transform_btn,
                position=Position.RIGHT_TOP,
                func=(widget._image_widget.dset_dlg.show_in_center_of_screen,),
            ),
            TutorialStep(
                title="Create mask",
                message="You can also create a mask withing image2image. These masks can be then used in"
                " <b>AutoIMS</b> or other software such as <b>QuPath</b>.",
                widget=widget.create_mask_btn,  # type: ignore[has-type]
                position=Position.RIGHT_TOP,
                func=(widget._image_widget.dset_dlg.hide,),
            ),
            TutorialStep(
                title="Generate masks",
                message="If you are planning on doing supervised learning in <b>AutoIMS</b>, you can generate"
                " compatible masks right within image2viewer!",
                widget=widget.export_mask_btn,  # type: ignore[has-type]
                position=Position.RIGHT_TOP,
            ),
            *_generic_statusbar(widget),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_crop_tutorial(widget: ImageCropWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2crop!",
                message="We would like to show you around before you get started!<br>This app let's you carve out"
                " regions of interest in images. This can be useful when you want to reduce the size of the microscopy"
                " images for e.g. image fusion. You can draw any shape on the canvas but we will use the bounding box"
                " to cut region out.",
                widget=widget.view.widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="Open previous project",
                message=OPEN_PROJECT,
                widget=widget._image_widget.import_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Control what should be shown",
                message=MANAGE_SELECTION,
                widget=widget._image_widget.more_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Add or remove image",
                message=ADD_IMAGES,
                widget=widget._image_widget.add_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Image crop",
                message="You can specify as many regions of interest as you wish. These are accessible by changing the"
                " `crop area` selection. Each time a new region is added, the list of available options updates."
                " The displayed values correspond to the bounding box around the region of interest.",
                widget=widget.index_choice,  # type: ignore[has-type]
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Preview",
                message="You can preview your selection by clicking here. This will extract each region of interest for"
                " each of the loaded images.",
                widget=widget.preview_crop_btn,  # type: ignore[has-type]
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Export to OME-TIFF",
                message="You can export each region of interest to specified directory. You will be prompted to select"
                " one.",
                widget=widget.crop_btn,  # type: ignore[has-type]
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Save project",
                message="You can save the current state of the viewer (loaded images, pixel size and transformation"
                " information) and reload it in the future without all that faffing about!",
                widget=widget._image_widget.export_btn,
                position=Position.RIGHT_TOP,
            ),
            *_generic_statusbar(widget),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_elastix_tutorial(widget: ImageElastixWindow) -> bool:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to the <b>Elastix</b> app!",
                message="We would like to show you around before you get started!<br>This app let's you register whole"
                "slide images using the <b>Elastix</b> framework. This app is heavily inspired by <b>napari-wsireg</b>"
                " that we've all been using, but it offers a couple of nice additions such as making it easier to"
                " initialize the registration process, improved image pre-processing, easier masking, better data"
                " organization and much more!",
                widget=widget.view.widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="Open previous project",
                message=OPEN_PROJECT,
                widget=widget._image_widget.import_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Add images",
                message=ADD_IMAGES,
                widget=widget._image_widget.add_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="More options",
                message=MORE_OPTIONS,
                widget=widget._image_widget.more_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Image list",
                message="Images that are to be registered will appear in this list. You can change the <b>modality</b>"
                " name, pixel size as well as adjust the pre-processing parameters that will govern how the"
                " registration is performed.",
                widget=widget.modality_list,
                position=Position.RIGHT_BOTTOM,
            ),
            TutorialStep(
                title="Change display type",
                message="You can toggle between viewing the <b>first</b> or <b>pre-processed</b> imaged by changing"
                " the value of this checkbox. Pre-processing can take a few seconds, depending on the pyramid"
                " level option.",
                widget=widget.use_preview_check,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Masking and cropping",
                message="If the registration is tricky, you can optionally mask or crop the image. Masking will ensure"
                " that only part of the image is used in registration (but the entire image will be exported)"
                " whereas cropping will crop the image first, and subsequently perform the registration.",
                widget=widget.mask_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Registration path",
                message="Similar to how napari-wsireg operates, you must define the <b>source</b> and <b>target</b>"
                " images. You can also optionally specify the <b>through</b> modality. The <b>source modality will be"
                " moved and therefore changed. The through modality can he helpful in aiding difficult registrations"
                " where the source and target modalities are not similar.",
                widget=widget.registration_settings,
                position=Position.RIGHT,
                func=(widget.registration_settings.expand,),
            ),
            TutorialStep(
                title="Export options",
                message="You can control how the data is exported by adjusting a few options hidden here.",
                widget=widget.hidden_settings,
                position=Position.RIGHT_BOTTOM,
                func=(widget.hidden_settings.expand,),
            ),
            TutorialStep(
                title="Project name",
                message="You can specify the name of the project here. This will be used when saving the project.",
                widget=widget.name_label,
                position=Position.RIGHT,
                func=(widget.registration_settings.collapse,),
            ),
            TutorialStep(
                title="Save",
                message="Click here to save the Elastix project to file.",
                widget=widget.save_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.hidden_settings.collapse,),
            ),
            TutorialStep(
                title="Open in viewer",
                message="Click here to open the <b>registered</b> project in the Viewer app (nothing will happen if the"
                " registration has not been performed).",
                widget=widget.viewer_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Register images",
                message="Click here to perform the image registration. There are a number of options available.",
                widget=widget.run_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Queue",
                message="You can see registrations tasks in the queue. Click here to open the queue view.",
                widget=widget.queue_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            *_generic_statusbar(widget),
        ]
    )
    tut.setFocus()
    tut.show()
    return True


def show_valis_tutorial(widget: ImageValisWindow) -> bool:
    """Show tutorial."""
    from qtextra.helpers import hyper
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    valis_ref = hyper("https://www.nature.com/articles/s41467-023-40218-9", value="Valis", prefix="")
    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to the <b>Valis</b> app!",
                message="We would like to show you around before you get started!<br>This app let's you register whole"
                "slide images using the <b>Valis</b> framework. Valis is a relatively new registration framework that"
                "performs multi-step registration using linear and non-linear transformations. Please see here for"
                f" more information {valis_ref}.",
                widget=widget.view.widget,
                position=Position.CENTER,
            ),
            TutorialStep(
                title="Open previous project",
                message=OPEN_PROJECT,
                widget=widget._image_widget.import_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Add images",
                message=ADD_IMAGES,
                widget=widget._image_widget.add_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="More options",
                message=MORE_OPTIONS,
                widget=widget._image_widget.more_btn,
                position=Position.RIGHT_TOP,
            ),
            TutorialStep(
                title="Image list",
                message="Images that are to be registered will appear in this list. You can change the <b>modality</b>"
                " name, pixel size as well as adjust the pre-processing parameters that will govern how the"
                " registration is performed.",
                widget=widget.modality_list,
                position=Position.RIGHT,
            ),
            TutorialStep(
                title="Change display type",
                message="You can toggle between viewing the <b>first</b> or <b>pre-processed</b> imaged by changing"
                " the value of this checkbox. Pre-processing can take a few seconds, depending on the pyramid"
                " level option.",
                widget=widget.use_preview_check,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Reference",
                message="You can optionally specify the reference image that will be used in the registration process. "
                " If one is not specified, it will be automatically determined based on similarity to other images.",
                widget=widget.reference_choice,
                position=Position.BOTTOM_RIGHT,
                func=(widget.registration_settings.expand,),
            ),
            TutorialStep(
                title="Export options",
                message="You can control how the data is exported by adjusting a few options hidden here.",
                widget=widget.hidden_settings,
                position=Position.RIGHT_BOTTOM,
                func=(widget.hidden_settings.expand,),
            ),
            TutorialStep(
                title="Project name",
                message="You can specify the name of the project here. This will be used when saving the project.",
                widget=widget.name_label,
                position=Position.RIGHT,
                func=(widget.registration_settings.collapse,),
            ),
            TutorialStep(
                title="Save",
                message="Click here to save the Elastix project to file.",
                widget=widget.save_btn,
                position=Position.BOTTOM_RIGHT,
                func=(widget.hidden_settings.collapse,),
            ),
            TutorialStep(
                title="Open in viewer",
                message="Click here to open the <b>registered</b> project in the Viewer app (nothing will happen if the"
                " registration has not been performed).",
                widget=widget.viewer_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Register images",
                message="Click here to perform the image registration. There are a number of options available.",
                widget=widget.run_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Queue",
                message="You can see registrations tasks in the queue. Click here to open the queue view.",
                widget=widget.queue_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            *_generic_statusbar(widget),
        ]
    )
    tut.setFocus()
    tut.show()
    return True
