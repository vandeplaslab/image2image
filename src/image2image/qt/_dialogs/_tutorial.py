"""Tutorial dialog."""
import typing as ty

if ty.TYPE_CHECKING:
    from image2image.qt.dialog_convert import ImageConvertWindow
    from image2image.qt.dialog_export import ImageExportWindow
    from image2image.qt.dialog_register import ImageRegistrationWindow
    from image2image.qt.dialog_viewer import ImageViewerWindow


def show_convert_tutorial(widget: "ImageConvertWindow") -> None:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to czi2tiff!",
                message="We would like to show you around before you get started!<br>This app allows you to convert"
                " Zeiss CZI images to OME-TIFF format. Sometimes, CZI image might contain multiple scenes which"
                " are not always supported by other software. This app lets you convert each scene to a OME-TIFF file"
                " which can be opened in other software.<br><br>Note. Some of the metadata might be lost during the"
                " process",
                widget=widget._image_widget,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="List of images",
                message="Here is a list of CZI images that will be converted to OME-TIFF format. If the CZI image has"
                " multiple scenes (no. scenes), each scene will be converted to a separate OME-TIFF file.",
                widget=widget.table,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Output directory",
                message="You can close the currently selected project by clicking here.",
                widget=widget.directory_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Convert to OME-TIFF",
                message="Click here to start the conversion process.",
                widget=widget.export_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Configuration panel",
                message="You can always stop conversion process by clicking here. The task will be stopped once the"
                " current step (e.g. scene) is finished.",
                widget=widget.export_btn.cancel_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.TOP_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()


def show_export_tutorial(widget: "ImageExportWindow") -> None:
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
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="List of images",
                message="Here are all the images that will be converted to Fusion CSV format.",
                widget=widget.table,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Output directory",
                message="You can close the currently selected project by clicking here.",
                widget=widget.directory_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Convert to CSV",
                message="Click here to start the conversion process.",
                widget=widget.export_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Configuration panel",
                message="You can always stop conversion process by clicking here. The task should stop almost"
                " immediately.",
                widget=widget.export_btn.cancel_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.TOP_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()


def show_register_tutorial(widget: "ImageRegistrationWindow") -> None:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2register!",
                message="We would like to show you around before you get started!<br>This app let's you generate"
                " image registration information between e.g. microscopy and IMS data.",
                widget=widget.view_fixed.widget,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="List of images",
                message="Here are all the images that will be converted to Fusion CSV format.",
                widget=widget.table,
                position=Position.BOTTOM,
            ),
            TutorialStep(
                title="Output directory",
                message="You can close the currently selected project by clicking here.",
                widget=widget.directory_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Convert to CSV",
                message="Click here to start the conversion process.",
                widget=widget.export_btn,
                position=Position.TOP,
            ),
            TutorialStep(
                title="Configuration panel",
                message="You can always stop conversion process by clicking here. The task should stop almost"
                " immediately.",
                widget=widget.export_btn.cancel_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.TOP_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()


def show_viewer_tutorial(widget: "ImageViewerWindow") -> None:
    """Show tutorial."""
    from qtextra.widgets.qt_tutorial import Position, QtTutorial, TutorialStep

    tut = QtTutorial(widget)
    tut.set_steps(
        [
            TutorialStep(
                title="Welcome to image2viewer!",
                message="We would like to show you around before you get started!<br>This app let's overlay multiple"
                " images (such as microscopy or IMS), apply appropriate image transformation and export as figures.",
                widget=widget.view.widget,
                position=Position.RIGHT,
            ),
            TutorialStep(
                title="Open previous project",
                message="If you've previously saved a project, you can open it here.",
                widget=widget.import_project_btn,
                position=Position.BOTTOM_RIGHT,
            ),
            TutorialStep(
                title="Control what should be shown",
                message="You can control what images should be loaded, which image channels should be displayed and how"
                " they should be spatially transformed.",
                widget=widget._image_widget,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Add or remove image",
                message="Add or remove images.",
                widget=widget._image_widget.add_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="More options",
                message="You can change the spatial pixel size or optionally extra ion images here.",
                widget=widget._image_widget.more_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Channel selection",
                message="Control which image channels should be shown or hidden. You can also use the layer list"
                " below which offers more options such as adjusting image contrast, changing opacity or colormap.",
                widget=widget._image_widget.channel_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Transform selection",
                message="When you load certain images (e.g. IMS), you might need to adjust the way it's shown in the "
                "canvas. Usually you've already co-registered your microscopy images but the IMS data won't lie in the"
                " right place as it has different pixel size, spatial orientation or position. You can load previous"
                " transformation data here and apply it to appropriate dataset.",
                widget=widget._image_widget.transform_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Generate masks",
                message="If you are planning on doing supervised learning in <b>AutoIMS</b>, you can generate compatible"
                " masks right within image2viewer!",
                widget=widget.export_mask_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Save project",
                message="You can save the current state of the viewer (loaded images, pixel size and transformation"
                " information) and reload it in the future without all that faffing about!",
                widget=widget.export_project_btn,
                position=Position.LEFT_TOP,
            ),
            TutorialStep(
                title="Screenshot to clipboard",
                message="You can take a screenshot of the canvas by clicking here! If you right-click on the button,"
                " a few extra options will be shown.",
                widget=widget.clipboard_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Save screenshot",
                message="You can save screenshot of the canvas by clicking here!. If you right-click on the button, "
                " a few extra options will be shown.",
                widget=widget.screenshot_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Show scalebar",
                message="You can show/hide scalebar on the canvas by clicking here. Make sure to set the right pixel"
                " size!.",
                widget=widget.scalebar_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Tutorial",
                message="If you wish to see this tutorial again at a future date, you can click here to show it.",
                widget=widget.tutorial_btn,
                position=Position.TOP_RIGHT,
            ),
            TutorialStep(
                title="Feedback",
                message="If you have some feedback, don't hesitate to send! You can do it directly in the app!",
                widget=widget.feedback_btn,
                position=Position.TOP_RIGHT,
            ),
        ]
    )
    tut.setFocus()
    tut.show()
