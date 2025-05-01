"""Registration paths."""

from __future__ import annotations

import typing as ty
from functools import partial

import qtextra.helpers as hp
from loguru import logger
from natsort import natsorted
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget

from image2image.config import get_elastix_config
from image2image.enums import REGISTRATION_PATH_HELP

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.elastix import ElastixReg

logger = logger.bind(src="RegistrationMap")


class RegistrationPaths(QWidget):
    """Widget for selecting registration paths."""

    evt_override = Signal()

    def __init__(self, parent: RegistrationMap):
        super().__init__(parent)
        self._parent: RegistrationMap = parent
        self.transformations = []
        self._init_ui()

    @property
    def registration_model(self) -> ElastixReg:
        """Registration model."""
        return self._parent.registration_model

    def _init_ui(self) -> None:
        self._choice = hp.make_combobox(
            self,
            options=[
                "rigid",
                "similarity",
                "affine",
                "nl",
                "nl2",
                "rigid_expanded",  # Expanded
                "similarity_expanded",
                "affine_expanded",
                "nl_expanded",
                "rigid_extreme",  # Extreme
                "similarity_extreme",
                "affine_extreme",
                "nl_extreme",
                "rigid_ams",  # AMS
                "similarity_ams",
                "affine_ams",
                "nl_ams",
                "rigid_anc",  # ANC
                "similarity_anc",
                "affine_anc",
                "nl_anc",
                "fi_correction",  # Other
                "nl_reduced",
                "nl_mid",
            ],
            default="rigid",
            tooltip="Select registration type(s)...",
        )
        self._choice.insertSeparator(self._choice.findText("nl2") + 1)  # after initials
        self._choice.insertSeparator(self._choice.findText("nl_expanded") + 1)  # after expanded
        self._choice.insertSeparator(self._choice.findText("nl_extreme") + 1)  # after extreme
        self._choice.insertSeparator(self._choice.findText("nl_ams") + 1)  # after AMC metrics
        self._choice.insertSeparator(self._choice.findText("nl_anc") + 1)  # after ANC metrics
        self._path = hp.make_label(
            self, "<please select transformations>", wrap=True, alignment=Qt.AlignmentFlag.AlignHCenter
        )

        self.auto_create_btn = hp.make_qta_btn(
            self,
            "magic",
            func=self.on_auto_create_paths,
            tooltip="Automatically create registration paths.",
            standout=True,
            normal=True,
        )

        layout = hp.make_form_layout(parent=self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        layout.addRow(
            hp.make_h_layout(
                hp.make_qta_label(self, "help", hover=True, tooltip=REGISTRATION_PATH_HELP),
                hp.make_qta_btn(
                    self,
                    "common",
                    func=self.on_select_from_common,
                    tooltip="Select from a list of commonly used transformation combinations.",
                    standout=True,
                    normal=True,
                ),
                hp.make_qta_btn(
                    self,
                    "replace",
                    func=self.evt_override.emit,
                    tooltip="Override existing transformations on each set of paths using the existing selection.",
                    standout=True,
                    normal=True,
                ),
                self.auto_create_btn,
                self._choice,
                hp.make_qta_btn(
                    self,
                    "add",
                    func=self.on_add_transformation,
                    tooltip="Add selected transformation to the list.",
                    standout=True,
                    normal=True,
                ),
                hp.make_qta_btn(
                    self,
                    "remove",
                    func=self.on_remove_transformation,
                    func_menu=self.on_reset_transformation,
                    tooltip="Remove last transformation from the list.\nRight-click to reset all transformations.",
                    standout=True,
                    normal=True,
                ),
                stretch_id=(4,),
                spacing=2,
                margin=(0, 0, 0, 0),
            )
        )
        layout.addRow(self._path)

    def on_auto_create_paths(self) -> None:
        """Automatically create registration paths."""
        menu = hp.make_menu(self)
        hp.make_menu_from_options(
            self,
            menu,
            ["direct (A -> B)", "cascade (A -> C via B)"],
            func=self._parent._on_create_paths,
        )
        hp.show_below_widget(menu, self.auto_create_btn)

    def on_select_from_common(self) -> None:
        """Select from list of common annotations."""
        menu = hp.make_menu(self)
        hp.make_menu_from_options(
            self,
            menu,
            [
                # linear only
                "rigid » affine",
                "rigid_expanded » affine_expanded",
                "rigid_extreme » affine_extreme",
                "rigid_extreme » affine_extreme » affine_extreme",
                None,
                # linear + non-linear
                "rigid » affine » nl",
                "rigid_expanded » affine_expanded » nl_expanded",
                "rigid_extreme » affine_extreme » nl_extreme",
                None,
                "affine » nl",
                "affine_expanded » nl_expanded",
                "affine_extreme » nl_extreme",
            ],
            func=self._on_select_from_common,
        )
        hp.show_below_widget(menu, self._choice, x_offset=-50)

    def _on_select_from_common(self, transformation: str) -> None:
        """Select from list of common annotations."""
        transformations = transformation.split(" » ")
        self.transformations = list(transformations)
        get_elastix_config().transformations = tuple(self.transformations)
        self._update_transformation_path()

    def on_add_transformation(self) -> None:
        """Add transformation to the list."""
        current = self._choice.currentText()
        self.transformations = list(self.transformations)  # ensure it's a list
        self.transformations.append(current)
        get_elastix_config().transformations = tuple(self.transformations)
        self._update_transformation_path()

    def on_remove_transformation(self) -> None:
        """Remove last transformation from the list."""
        self.transformations = list(self.transformations)
        if self.transformations:
            self.transformations.pop()
            get_elastix_config().transformations = tuple(self.transformations)
            self._update_transformation_path()

    def on_reset_transformation(self) -> None:
        """Reset transformation list."""
        self.transformations = []
        get_elastix_config().transformations = tuple(self.transformations)
        self._update_transformation_path()

    def _update_transformation_path(self) -> None:
        """Update transformation path."""
        self._path.setText(
            " » ".join(self.transformations) if self.transformations else "<please select transformations>"
        )
        self._path.setToolTip(
            "<br>".join(self.transformations) if self.transformations else "<please select transformations>"
        )

    @property
    def registration_paths(self) -> list[str]:
        """Registration paths."""
        return self.transformations

    @registration_paths.setter
    def registration_paths(self, value: str | list[str]):
        if isinstance(value, str):
            value = [value]
        self.transformations = list(value)
        self._update_transformation_path()


class RegistrationMap(QWidget):
    """Registration map."""

    evt_message = Signal(str)
    evt_valid = Signal(bool, list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._parent = parent
        self.transformation_map: dict[str, list[str | None]] = {}
        self._init_ui()

    @property
    def registration_model(self) -> ElastixReg:
        """Registration model."""
        return self._parent.registration_model

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
        self._registration_path.evt_override.connect(self.on_override_paths)
        self._registration_path.registration_paths = get_elastix_config().transformations

        layout = hp.make_form_layout(parent=self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addRow(self._registration_path)
        layout.addRow(
            hp.make_label(self, "Src", tooltip="Source modality (moving)"),
            hp.make_h_layout(
                self._source_choice,
                hp.make_qta_btn(
                    self,
                    "previous",
                    tooltip="Show previous ion image.",
                    normal=True,
                    func=partial(self.on_increment_source, -1),
                    standout=True,
                ),
                hp.make_qta_btn(
                    self,
                    "next",
                    tooltip="Show next ion image.",
                    normal=True,
                    func=partial(self.on_increment_source, 1),
                    standout=True,
                ),
                spacing=1,
                margin=0,
            ),
        )
        layout.addRow(
            hp.make_label(self, "Thr", tooltip="Through modality (in-between)"),
            hp.make_h_layout(
                self._through_choice,
                hp.make_qta_btn(
                    self,
                    "previous",
                    tooltip="Show previous ion image.",
                    normal=True,
                    func=partial(self.on_increment_through, -1),
                    standout=True,
                ),
                hp.make_qta_btn(
                    self,
                    "next",
                    tooltip="Show next ion image.",
                    normal=True,
                    func=partial(self.on_increment_through, 1),
                    standout=True,
                ),
                spacing=1,
                margin=0,
            ),
        )
        layout.addRow(
            hp.make_label(self, "Tgt", tooltip="Target modality (fixed)"),
            hp.make_h_layout(
                self._target_choice,
                hp.make_qta_btn(
                    self,
                    "previous",
                    tooltip="Show previous ion image.",
                    normal=True,
                    func=partial(self.on_increment_target, -1),
                    standout=True,
                ),
                hp.make_qta_btn(
                    self,
                    "next",
                    tooltip="Show next ion image.",
                    normal=True,
                    func=partial(self.on_increment_target, 1),
                    standout=True,
                ),
                spacing=1,
                margin=0,
            ),
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(
                    self,
                    "Add path",
                    func=self.on_add_path,
                    tooltip="Add current registration path to the list.",
                ),
                hp.make_btn(
                    self,
                    "Remove path",
                    func=self.on_remove_path,
                    tooltip="Remove current registration path from the list.",
                ),
                hp.make_btn(
                    self,
                    "Reset",
                    func=self.on_reset_paths,
                    tooltip="Remove all registration paths. You will be asked to confirm.",
                ),
                spacing=2,
                margin=0,
                stretch_before=True,
                stretch_after=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "Paths"),
                self._choice,
                hp.make_qta_btn(
                    self, "graph", tooltip="Preview paths", func=self.on_preview, standout=True, normal=True
                ),
                hp.make_qta_btn(
                    self, "reload", tooltip="Refresh paths", func=self.populate, standout=True, normal=True
                ),
                spacing=2,
                stretch_id=(1,),
            )
        )
        # layout.addRow(self._warning_label)

    def on_increment_source(self, step: int) -> None:
        """Increment source."""
        hp.increment_combobox(self._source_choice, step)

    def on_increment_target(self, step: int) -> None:
        """Increment target."""
        hp.increment_combobox(self._target_choice, step)

    def on_increment_through(self, step: int) -> None:
        """Increment through."""
        hp.increment_combobox(self._through_choice, step)

    def on_preview(self) -> None:
        """Preview paths as network."""
        from networkx.exception import NetworkXNoPath

        from image2image.qt._wsi._network import NetworkViewer

        try:
            dlg = NetworkViewer(self._parent)
            dlg.show()
        except NetworkXNoPath as exc:
            hp.toast(
                hp.get_main_window(),
                "Registration path is not valid",
                f"Registration paths re not valid.<br>Encountered an error: {exc}",
                duration=5000,
                icon="error",
                position="top_left",
            )

    def on_path_choice(self, _: ty.Any = None) -> None:
        """Handle path selection."""
        registration_model: ElastixReg = self.registration_model
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

    def _get_registration_path_data(self) -> tuple[bool, str, str, str | None, str, str, str, str]:
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

    def on_image_choice(self, _: ty.Any = None) -> None:
        """Handle image selection."""
        valid, source, target, through, source_on, target_on, through_on, _ = self._get_registration_path_data()
        hp.set_object_name(self._source_choice, object_name=source_on)
        hp.set_object_name(self._target_choice, object_name=target_on)
        hp.set_object_name(self._through_choice, object_name=through_on)
        if any(v != "" for v in [source, target, through]):
            self._choice.setCurrentText("")

    def _log_message(self, message: str) -> None:
        """Log message."""
        logger.trace(message)
        self.evt_message.emit(message)

    def on_override_paths(self) -> None:
        """Overwrite paths."""
        registrations = self._registration_path.registration_paths
        if not registrations:
            return
        registration_model: ElastixReg = self.registration_model
        if not registration_model or not hp.confirm(
            self, "Are you sure you want to overwrite all registration paths?", "Please confirm."
        ):
            return
        for node in registration_model.registration_nodes:
            node["params"] = registrations
        self.populate_paths()
        self.toggle_name()
        self._log_message("Overwritten all registration paths.")

    def _on_create_paths(self, option: str) -> None:
        """Automatically create paths."""
        names = natsorted(self.registration_model.get_image_modalities(with_attachment=False))
        if not names or len(names) <= 1:
            return

        target = hp.choose_from_list(
            self,
            names,
            title="Select target modality",
            text="Please select the <b>target</b> modality.<br>This modality will be used as the fixed image.",
            multiple=False,
        )
        if not target:
            return
        if not self.on_reset_paths():
            return

        kind = "cascade" if "cascade" in option else "direct"
        target_index = names.index(target)

        # direct registration is very simple, you go from A -> B, C -> B, etc
        if kind == "direct":
            for source in names:
                if source == target:
                    continue
                self.registration_model.add_registration_path(
                    source=source, target=target, transform=self._registration_path.registration_paths
                )
        # cascade registration is more complex because you must establish the through image
        else:
            indices = list(range(len(names)))
            indices_before = indices[0:target_index]
            indices_after = indices[target_index + 1 :]
            for source_index in indices_before:
                source = names[source_index]
                # if index is immediately before target, then we don't have through modality
                through = None if source_index == target_index - 1 else names[source_index + 1]
                self.registration_model.add_registration_path(
                    source=source, target=target, through=through, transform=self._registration_path.registration_paths
                )
            for source_index in reversed(indices_after):
                source = names[source_index]
                # if index is immediate after target, then we don't have through modality
                through = None if source_index == target_index + 1 else names[source_index - 1]
                self.registration_model.add_registration_path(
                    source=source, target=target, through=through, transform=self._registration_path.registration_paths
                )
        self.populate_paths()
        self.toggle_name()
        self.validate()

    def on_add_path(self) -> None:
        """Add path."""
        valid, source, target, through, *_, path = self._get_registration_path_data()
        if not source and not target:
            return
        registrations = self._registration_path.registration_paths
        if not registrations:
            return
        if not valid:
            hp.warn(self, "Please select source and target images.")
            return
        # self._warning_label.setText("")
        registration_model: ElastixReg = self.registration_model
        if not registration_model.has_registration_path(source, target, through):
            registration_model.add_registration_path(source, target, through=through, transform=registrations)
            self._log_message(f"Added registration path: {source} » {through} » {target}")
        else:
            registration_model.update_registration_path(source, target, through=through, transform=registrations)
            self._log_message(f"Updated registration path: {source} » {through} » {target}")
        self.populate_paths()
        self._choice.setCurrentText(path)
        self.toggle_name()
        self.validate()

    def on_remove_path(self) -> None:
        """Add path."""
        valid, source, target, through, *_ = self._get_registration_path_data()
        if not valid:
            logger.warning("Could not remove path.")
            # self._warning_label.setText("Please select source and target images.")
            return
        registration_model: ElastixReg = self.registration_model
        registration_model.remove_registration_path(source, target, through)
        self.populate_paths()
        self.toggle_name()
        self.validate()
        self._log_message(f"Removed registration path: {source} » {through} » {target}")

    def on_reset_paths(self, force: bool = False) -> bool:
        """Reset all registration paths."""
        if self.registration_model.n_registrations == 0:
            return True
        if force or hp.confirm(self, "Are you sure you want to reset all registration paths?", "Please confirm."):
            registration_model: ElastixReg = self.registration_model
            registration_model.reset_registration_paths()
            self.populate_paths()
            self.toggle_name()
            self.validate()
            self._log_message("Reset all registration paths.")
            return True
        return False

    def toggle_name(self) -> None:
        """Toggle name."""
        self._parent.modality_list.toggle_name(self.registration_model.n_registrations > 0)

    def populate(self) -> None:
        """Populate options."""
        self.populate_images()
        self.populate_paths()
        self.toggle_name()
        self.validate()

    def depopulate(self) -> None:
        """Populate options."""
        self.populate_images()
        self.populate_paths()
        self.toggle_name()
        self.validate()

    def validate(self) -> None:
        """Validate paths."""
        is_valid, errors = self.registration_model.validate_paths(log=False)
        self.evt_valid.emit(is_valid, errors)

    def populate_images(self) -> None:
        """Populate options."""
        registration_model: ElastixReg = self._parent.registration_model
        # add available modalities
        options = [""] + natsorted(registration_model.get_image_modalities(with_attachment=False))
        hp.combobox_setter(self._source_choice, items=options, set_item=self._source_choice.currentText())
        hp.combobox_setter(self._target_choice, items=options, set_item=self._target_choice.currentText())
        hp.combobox_setter(self._through_choice, items=options, set_item=self._through_choice.currentText())

    def populate_paths(self) -> None:
        """Populate options."""
        registration_model: ElastixReg = self._parent.registration_model
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
        paths = natsorted(set(paths))
        hp.combobox_setter(self._choice, items=paths, set_item=self._choice.currentText())
