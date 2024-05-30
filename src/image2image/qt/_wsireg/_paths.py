"""Registration paths."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from loguru import logger
from qtpy.QtWidgets import QWidget

from image2image.config import CONFIG

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.iwsireg import IWsiReg

logger = logger.bind(src="RegistrationMap")


class RegistrationPaths(QWidget):
    """Widget for selecting registration paths."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.transformations = []
        self._init_ui()

    def _init_ui(self) -> None:
        self._choice = hp.make_combobox(
            self,
            options=[
                "rigid",
                "affine",
                "similarity",
                "nl",
                "fi_correction",
                "nl_reduced",
                "nl_mid",
                "nl2",
                "rigid_expanded",
                "affine_expanded",
                "nl_expanded",
                "rigid_ams",
                "affine_ams",
                "nl_ams",
                "rigid_anc",
                "affine_anc",
                "similarity_anc",
                "nl_anc",
            ],
            default="rigid",
            tooltip="Select registration type(s)...",
        )
        self._path = hp.make_label(self, "<please select transformations>", wrap=True)

        layout = hp.make_form_layout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addRow(
            hp.make_h_layout(
                self._choice,
                hp.make_btn(self, "Add", func=self.on_add_transformation),
                hp.make_btn(self, "Reset", func=self.on_reset_transformation),
                stretch_id=(0,),
                spacing=2,
                margin=(0, 0, 0, 0),
            )
        )
        layout.addRow(self._path)

    def on_add_transformation(self) -> None:
        """Add transformation to the list."""
        current = self._choice.currentText()
        self.transformations.append(current)
        CONFIG.transformations = tuple(self.transformations)
        self._update_transformation_path()

    def on_reset_transformation(self) -> None:
        """Reset transformation list."""
        self.transformations = []
        CONFIG.transformations = tuple(self.transformations)
        self._update_transformation_path()

    def _update_transformation_path(self) -> None:
        """Update transformation path."""
        self._path.setText(
            " » ".join(self.transformations) if self.transformations else "<please select transformations>"
        )

    @property
    def registration_paths(self) -> list[str]:
        """Registration paths."""
        return self.transformations

    @registration_paths.setter
    def registration_paths(self, value: str | list[str]):
        if isinstance(value, str):
            value = [value]
        self.transformations = value
        self._update_transformation_path()


class RegistrationMap(QWidget):
    """Registration map."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._parent = parent
        self.transformation_map: dict[str, list[str | None]] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        """Create UI."""
        self._choice = hp.make_combobox(
            self, options=[], tooltip="Select already defined paths...", func=self.on_path_choice
        )
        self._source_choice = hp.make_combobox(
            self,
            options=[],
            tooltip="Select source image. This image will be moving. If a through image is also selected, then the"
            "\nthrough the source image will be registered to the through image and registration of"
            "\nthrough > target will be stacked with the source > through.",
            func=self.on_image_choice,
        )
        self._target_choice = hp.make_combobox(
            self, options=[], tooltip="Select target image. This image will be fixed.", func=self.on_image_choice
        )
        self._through_choice = hp.make_combobox(
            self,
            options=[],
            tooltip="Select through image. The source image will be registered to the through image and the"
            "\nthrough image is assumed to be registered to the target image. This gives an indirect registration.",
            func=self.on_image_choice,
        )

        self._registration_path = RegistrationPaths(self)
        self._registration_path.registration_paths = CONFIG.transformations
        self._warning_label = hp.make_label(self, "", color="warning", wrap=True)
        self._warning_label.setVisible(False)

        layout = hp.make_form_layout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addRow(self._registration_path)
        layout.addRow(hp.make_label(self, "Source"), self._source_choice)
        layout.addRow(hp.make_label(self, "Target"), self._target_choice)
        layout.addRow(hp.make_label(self, "Through (optional)"), self._through_choice)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Add path", func=self.on_add_path),
                hp.make_btn(self, "Remove path", func=self.on_remove_path),
                hp.make_btn(self, "Reset", func=self.on_reset_paths),
                spacing=2,
                margin=0,
                stretch_before=True,
                stretch_after=True,
            )
        )
        layout.addRow(hp.make_label(self, "Registration paths"), self._choice)
        layout.addRow(self._warning_label)

    def on_path_choice(self, _=None) -> None:
        """Handle path selection."""
        registration_model: IWsiReg = self._parent.registration_model
        path = self._choice.currentText()
        if path:
            parts = path.split(" » ")
            source, target, through = parts[0], parts[-1], parts[1] if len(parts) == 3 else None
            self._source_choice.setCurrentText(source)
            self._target_choice.setCurrentText(target)
            self._through_choice.setCurrentText(through or "")
            index = registration_model.find_index_of_registration_path(source, target, through)
            if index is not None:
                node = registration_model.registration_nodes[index]
                self._registration_path.registration_paths = node["params"]

    def _get_registration_path_data(self):
        """Check registration path."""
        source = self._source_choice.currentText()
        target = self._target_choice.currentText()
        through = self._through_choice.currentText()
        source_on = "error" if not source else ""
        source_on = "error" if source and target and (source == target) else source_on
        target_on = "error" if not target else ""
        target_on = "error" if source and target and (source == target) else target_on
        through_on = "error" if through and (through in (source, target)) else ""
        path = f"{source} » {target}" if through == "" else f"{source} » {through} » {target}"
        valid = not any(v != "" for v in [source_on, target_on, through_on])
        return valid, source, target, through or None, source_on, target_on, through_on, path

    def on_image_choice(self, _=None) -> None:
        """Handle image selection."""
        valid, source, target, through, source_on, target_on, through_on, _ = self._get_registration_path_data()
        hp.set_object_name(self._source_choice, object_name=source_on)
        hp.set_object_name(self._target_choice, object_name=target_on)
        hp.set_object_name(self._through_choice, object_name=through_on)
        if any(v != "" for v in [source, target, through]):
            self._choice.setCurrentText("")

    def on_add_path(self) -> None:
        """Add path."""
        valid, source, target, through, *_, path = self._get_registration_path_data()
        if not source and not target:
            self._warning_label.setText("Please select source and target images.")
            return
        registrations = self._registration_path.registration_paths
        if not registrations:
            self._warning_label.setText("Please select registration path.")
            return
        if not valid:
            hp.warn(self, "Please select source and target images.")
            return
        self._warning_label.setText("")
        registration_model: IWsiReg = self._parent.registration_model
        if not registration_model.has_registration_path(source, target, through):
            registration_model.add_registration_path(source, target, transform=registrations, through=through)
            logger.trace(f"Added registration path: {source} » {through} » {target}")
            self.populate_paths()
            self._choice.setCurrentText(path)
        self._parent.modality_list.toggle_name(registration_model.n_registrations > 0)

    def on_remove_path(self) -> None:
        """Add path."""
        valid, source, target, through, *_ = self._get_registration_path_data()
        if not valid:
            self._warning_label.setText("Please select source and target images.")
            return
        registration_model: IWsiReg = self._parent.registration_model
        registration_model.remove_registration_path(source, target, through)
        logger.trace(f"Removed registration path: {source} » {through} » {target}")
        self.populate_paths()
        self._parent.modality_list.toggle_name(registration_model.n_registrations > 0)

    def on_reset_paths(self) -> None:
        """Reset all registration paths."""
        if not hp.confirm(self, "Are you sure you want to reset all registration paths?", "Please confirm."):
            return
        registration_model: IWsiReg = self._parent.registration_model
        registration_model.reset_registration_paths()
        self.populate_paths()

    def populate(self) -> None:
        """Populate options."""
        self.populate_images()
        self.populate_paths()

    def depopulate(self) -> None:
        """Populate options."""
        self.populate_images()
        self.populate_paths()

    def populate_images(self) -> None:
        """Populate options."""
        registration_model: IWsiReg = self._parent.registration_model
        # add available modalities
        options = [""] + list(registration_model.modalities.keys())
        hp.combobox_setter(self._source_choice, items=options, set_item=self._source_choice.currentText())
        hp.combobox_setter(self._target_choice, items=options, set_item=self._target_choice.currentText())
        hp.combobox_setter(self._through_choice, items=options, set_item=self._through_choice.currentText())

    def populate_paths(self) -> None:
        """Populate options."""
        registration_model: IWsiReg = self._parent.registration_model
        # add available paths
        paths = [""]
        for node in registration_model.registration_nodes:
            modalities = node["modalities"]
            source = modalities["source"]
            targets = registration_model.registration_paths[source]
            if len(targets) == 1:
                paths.append(f"{source} » {targets[0]}")
            else:
                paths.append(f"{source} » {targets[0]} » {targets[1]}")
        paths = set(paths)
        hp.combobox_setter(self._choice, items=paths, set_item=self._choice.currentText())
