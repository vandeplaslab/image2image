"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from loguru import logger
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.table_config import TableConfig
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QDialog, QHeaderView, QMenuBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

import image2image.constants as C
from image2image import __version__
from image2image.config import STATE, get_merge_config
from image2image.enums import ALLOWED_IMAGE_FORMATS_TIFF_ONLY
from image2image.qt._dialog_mixins import NoViewerMixin
from image2image.qt._dialogs._select import LoadWidget
from image2image.utils.utilities import format_shape, log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


def get_metadata(
    readers_metadata: dict[Path, dict[int, dict[str, list[bool | int | str]]]],
) -> dict[Path, dict[int, dict[str, list[int | str]]]]:
    """Cleanup metadata."""
    metadata = {}
    for path, reader_metadata in readers_metadata.items():
        metadata_ = {}
        for scene_index, scene_metadata in reader_metadata.items():
            name = scene_metadata["name"]
            channel_name_to_ids = {}
            # iterate over channels and merge if necessary
            for index, channel_id in enumerate(scene_metadata["channel_ids"]):
                if scene_metadata["keep"][index]:
                    channel_name_to_ids[scene_metadata["channel_names"][index]] = channel_id

            # cleanup by removing any duplicates and sorting indices
            for channel_name in channel_name_to_ids:
                if isinstance(channel_name_to_ids[channel_name], list):
                    channel_name_to_ids[channel_name] = sorted(set(channel_name_to_ids[channel_name]))
            channel_names = list(channel_name_to_ids.keys())
            channel_ids = list(channel_name_to_ids.values())
            metadata_[scene_index] = {"name": name, "channel_ids": channel_ids, "channel_names": channel_names}
        metadata[path] = metadata_
    return metadata


class ImageMergeWindow(NoViewerMixin):
    """Image viewer dialog."""

    APP_NAME = "merge"

    worker: GeneratorWorker | None = None

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("key", "key", "str", 0)
        .add("name", "name", "str", 0)
        .add("pixel size (um)", "resolution", "str", 0)
        .add("image size (px)", "image_size", "str", 0)
        .add("scenes & channels", "metadata", "str", 0)
    )

    _get_metadata = staticmethod(get_metadata)

    def __init__(self, parent: QWidget | None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG = get_merge_config()
        super().__init__(
            parent,
            f"image2image: Merge images (v{__version__})",
            run_check_version=run_check_version,
        )
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self.reader_metadata: dict[Path, dict[int, dict[str, list[bool | int | str]]]] = {}
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.auto_pyramid = False
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        READER_CONFIG.split_rgb = True
        READER_CONFIG.only_last_pyramid = False

    def on_cleanup_reader_metadata(self, model: DataModel) -> None:
        """Cleanup metadata."""
        if not model:
            return
        paths = model.paths
        to_remove = []
        for path in self.reader_metadata:
            if path not in paths:
                to_remove.append(path)
        for path in to_remove:
            del self.reader_metadata[path]
            logger.trace(f"Removed metadata for {path}")

    def on_populate_table(self) -> None:
        """Load data."""
        self.on_depopulate_table()
        wrapper = self.data_model.wrapper
        if wrapper:
            for reader in wrapper.reader_iter():
                index = hp.find_in_table(self.table, self.TABLE_CONFIG.key, reader.key)
                if index is not None:
                    continue

                # get model information
                index = self.table.rowCount()
                reader_metadata = self.reader_metadata.get(reader.path, {})

                self.table.insertRow(index)

                # add name item
                table_item = QTableWidgetItem(reader.key)
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.key, table_item)

                # add name item
                table_item = QTableWidgetItem(reader.clean_name)
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.name, table_item)

                # add resolution item
                table_item = QTableWidgetItem(f"{reader.resolution:.2f}")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.resolution, table_item)

                # add type item
                if reader.reader_type == "image":
                    shape = reader.shape
                    image_size = format_shape(shape)
                else:
                    image_size = "N/A"
                type_item = QTableWidgetItem(image_size)
                type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                type_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.image_size, type_item)

                table_item = QTableWidgetItem("")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, self.TABLE_CONFIG.metadata, table_item)

                if reader_metadata:
                    self.reader_metadata[reader.path] = reader_metadata
                else:
                    self.reader_metadata[reader.path] = {}
                    for scene_index in range(reader.n_scenes):
                        self.reader_metadata[reader.path][scene_index] = {
                            "key": reader.key,
                            "keep": [True] * reader.n_channels,
                            "channel_ids": reader.channel_ids,
                            "channel_names": reader.channel_names,
                            "name": reader.clean_name,
                        }
        self.on_update_reader_metadata()

    def on_merge(self) -> None:
        """Process data."""
        from image2image_io.writers import merge_images

        if self.output_dir is None:
            hp.warn_pretty(self, "No output directory was selected. Please select directory where to save data.")
            return

        paths, image_shapes, pixel_sizes = [], [], []
        for row in range(self.table.rowCount()):
            key = self.table.item(row, self.TABLE_CONFIG.key).text()
            reader = self.data_model.get_reader_for_key(key)
            if reader is None:
                logger.warning(f"Could not find path for {key}")
                continue
            if reader:
                paths.append(reader.path)
                image_shapes.append(reader.image_shape)
                pixel_sizes.append(f"{reader.resolution:.5f}")

        name = self.name_edit.text().strip()

        if not name:
            hp.warn_pretty(self, "No name was provided. Please provide a name for the merged image.")
            return
        if not paths:
            hp.warn_pretty(self, "No images to merge. Please load images to merge.")
            return
        if len(paths) < 2:
            hp.warn_pretty(self, "At least two images are required to merge.")
            return
        if len(set(image_shapes)) > 1:
            hp.warn_pretty(self, "Images must have the same shape to merge.")
            return
        if len(set(pixel_sizes)) > 1:
            hp.warn_pretty(self, "Images must have the same pixel size to merge.")
            return

        # get metadata and add extra information such as the new tag/name
        metadata = get_metadata(self.reader_metadata)
        # # update metadata with the name
        # for path in metadata:
        #     reader = self.data_model.get_reader(path)
        #     row = hp.find_in_table(self.table, self.TABLE_CONFIG.key, reader.key)
        #     name_for_path = self.table.item(row, self.TABLE_CONFIG.name).text()
        #     metadata[path]["name"] = name_for_path

        output_dir = self.output_dir
        if paths and STATE.is_mac_arm_pyinstaller:
            logger.warning("Merging process is running in the UI thread, meaning that the app will freeze!")
            merge_images(
                name=name,
                paths=paths,
                output_dir=output_dir,
                as_uint8=self.CONFIG.as_uint8,
                tile_size=self.CONFIG.tile_size,
                metadata=metadata,
                overwrite=self.CONFIG.overwrite,
            )
            self._on_export_finished()
        else:
            self.worker = create_worker(
                merge_images,
                name=name,
                paths=paths,
                output_dir=output_dir,
                as_uint8=self.CONFIG.as_uint8,
                tile_size=self.CONFIG.tile_size,
                metadata=metadata,
                overwrite=self.CONFIG.overwrite,
                _start_thread=True,
                _connect={
                    "finished": self._on_export_finished,
                    "errored": self._on_export_error,
                },
            )
            hp.disable_widgets(self.export_btn.active_btn, disabled=True)
            self.export_btn.active = True

    def on_cancel(self) -> None:
        """Cancel processing."""
        if self.worker:
            self.worker.quit()
            logger.trace("Requested aborting of the export process.")

    @ensure_main_thread()  # type: ignore[misc]
    def _on_export_error(self, exc: Exception) -> None:
        """Failed exporting of the CSV."""
        self.on_toggle_export_btn(force=True)
        log_exception_or_error(exc)
        self.worker = None

    @ensure_main_thread()
    def _on_export_finished(self):
        """Failed exporting of the CSV."""
        self.on_toggle_export_btn(force=True)
        self.worker = None

    def on_toggle_export_btn(self, force: bool = False) -> None:
        """Toggle export button."""
        disabled = False
        hp.disable_widgets(self.export_btn.active_btn, disabled=disabled)
        self.export_btn.active = disabled

    def on_select(self, evt) -> None:
        """Select channels."""
        from image2image.qt._dialogs._rename import ChannelRenameDialog

        row = evt.row()
        column = evt.column()
        name = self.table.item(row, self.TABLE_CONFIG.key).text()
        reader = self.data_model.get_reader_for_key(name)
        scene_metadata = self.reader_metadata[reader.path][reader.scene_index]
        if column == self.TABLE_CONFIG.metadata:
            dlg = ChannelRenameDialog(self, reader.scene_index, scene_metadata, allow_merge=False)
            if dlg.exec_() == QDialog.DialogCode.Accepted:
                self.reader_metadata[reader.path][reader.scene_index] = dlg.scene_metadata
                self.on_update_reader_metadata()
        elif column == self.TABLE_CONFIG.name:
            name = self.table.item(row, self.TABLE_CONFIG.name).text()
            new_name = hp.get_text(self, "Rename image", "Enter new name for the image.", name)
            if new_name and new_name != name:
                self.table.item(row, self.TABLE_CONFIG.name).setText(new_name)
                self.reader_metadata[reader.path][reader.scene_index]["name"] = new_name

    def on_check_file(self) -> None:
        """Check whether file already exists."""
        if self.output_dir:
            name = self.name_edit.text().strip()
            name = name.replace(".ome", "").replace(".tiff", "").replace(".tif", "")
            filename = self.output_dir / f"{name}.ome.tiff"
            hp.update_widget_style(self.name_edit, "warning" if filename.exists() else "")

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update configuration file."""
        self.CONFIG.update(
            as_uint8=self.as_uint8.isChecked(),
            overwrite=self.overwrite.isChecked(),
            tile_size=int(self.tile_size.currentText()),
        )

    def _setup_ui(self):
        """Create panel."""
        self._image_widget = LoadWidget(
            self, None, self.CONFIG, allow_channels=False, available_formats=ALLOWED_IMAGE_FORMATS_TIFF_ONLY
        )
        self._image_widget.dset_dlg.evt_closed.connect(self.on_cleanup_reader_metadata)

        columns = self.TABLE_CONFIG.to_columns()
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(columns))  # name, progress, key
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setCornerButtonEnabled(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(True)
        self.table.doubleClicked.connect(self.on_select)

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.key, QHeaderView.ResizeMode.Stretch)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.resolution, QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.image_size, QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.metadata, QHeaderView.ResizeMode.ResizeToContents)
        vertical_header = self.table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.name_edit = hp.make_line_edit(
            self,
            "",
            placeholder="Name to be used for merged images.",
            tooltip="Name of the merged image.",
            func_changed=self.on_check_file,
            object_name="warning",
        )
        self.tile_size = hp.make_combobox(
            self,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{self.CONFIG.tile_size}",
            func=self.on_update_config,
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip=C.UINT8_TIP,
            checked=True,
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )
        self.overwrite = hp.make_checkbox(
            self,
            "Overwrite existing files",
            tooltip="Overwrite existing files without having to delete them (e.g. if adding merged channels).",
            checked=True,
            value=self.CONFIG.overwrite,
            func=self.on_update_config,
        )

        self.directory_btn = hp.make_btn(
            self,
            "Set output directory...",
            tooltip="Specify output directory for images...",
            func=self.on_set_output_dir,
        )
        self.output_dir_label = hp.make_label(
            self, f"<b>Output directory</b>: {hp.hyper(self.output_dir)}", enable_url=True
        )
        self.export_btn = hp.make_active_progress_btn(
            self,
            "Merge to OME-TIFF",
            tooltip="Merge to OME-TIFF...",
            func=self.on_merge,
            cancel_func=self.on_cancel,
        )

        side_layout = hp.make_form_layout()
        side_layout.addRow(
            hp.make_label(
                self,
                "This app will <b>merge</b> multiple images into a single <b>pyramidal OME-TIFF</b>.<br>"
                " Please ensure that the image shape and resolution (pixel size) match across the images.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="large_text",
                enable_url=True,
                wrap=True,
            )
        )
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(self.table)
        side_layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on the <b>name</b> field to rename the image.<br>"
                "<b>Tip.</b> Double-click on the <b>scenes & channels</b> field to select/deselect scenes/channels"
                " or rename channel names.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
                wrap=True,
            )
        )
        side_layout.addRow(hp.make_h_line(self))
        side_layout.addRow(self.name_edit)
        side_layout.addRow(self.directory_btn)
        side_layout.addRow(self.output_dir_label)
        side_layout.addRow("Tile size", hp.make_h_layout(self.tile_size, stretch_after=True))
        side_layout.addRow(
            "Reduce file size",
            hp.make_h_layout(
                self.as_uint8,
                hp.make_warning_label(
                    self,
                    "While this option reduces the amount of space an image takes on your disk, it can lead to data"
                    " loss and should be used with caution.",
                    normal=True,
                ),
                spacing=2,
                stretch_after=True,
            ),
        )
        side_layout.addRow("Overwrite", hp.make_h_layout(self.overwrite, stretch_after=True))
        side_layout.addRow(self.export_btn)

        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(side_layout)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.tiff, .scn, .svs, .tif, .ndpi, .qptiff, .qptiff.raw, .qptiff.intermediate)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu().menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update({"data_model": self.data_model, "wrapper": self.data_model.wrapper})
        return variables

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(self, "confirm_close", config=self.CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(self, "confirm_close", config=self.CONFIG).exec_() != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_merge_tutorial

        if show_merge_tutorial(self):
            self.CONFIG.update(first_time=False)

    def dropEvent(self, event: QDropEvent) -> None:
        """Drop event."""
        self._setup_config()
        super().dropEvent(event)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="merge", level=0)
