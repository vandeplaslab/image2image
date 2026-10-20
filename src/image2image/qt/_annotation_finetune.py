"""Explicit model and optimization choices for annotation-based instance fine-tuning."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import qtextra.helpers as hp
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QSizePolicy

from image2image.models.annotation import FineTuneSettings
from image2image.utils.annotation.ml import backend_available
from image2image.utils.annotation.model_cache import model_cache_directory, pretrained_model_help

if TYPE_CHECKING:
    from image2image.qt.dialog_annotate import ImageAnnotationWindow


def model_requirements_html(backend: str, entries: list[dict[str, Any]]) -> str:
    """Explain unavailable architectures beside a model selector, including setup documentation."""
    blocked = [entry for entry in entries if entry.get("reason")]
    if not blocked:
        return ""
    families: dict[str, list[str]] = {}
    for entry in blocked:
        families.setdefault(entry["reason"], []).append(entry.get("display_name", entry["name"]))
    message = "<br>".join(f"<b>{escape(', '.join(names))}</b>: {escape(reason)}" for reason, names in families.items())
    url = (
        "https://computational-cell-analytics.github.io/micro-sam/#installation"
        if backend == "micro-sam"
        else "https://cellpose.readthedocs.io/en/latest/_modules/cellpose/models.html"
    )
    return message + "<br>" + hp.hyper(url, value="Backend installation instructions")


class FineTuneDialog(QDialog):
    """Choose an available base checkpoint and inspect instance-training assumptions before starting."""

    def __init__(self, parent: ImageAnnotationWindow, class_ids: tuple[str, ...]) -> None:
        super().__init__(parent)
        self.window, self.class_ids = parent, class_ids
        self.setWindowTitle("Fine-tune instance model")
        self.resize(620, 780)
        layout = hp.make_v_layout(parent=self)
        panel, scroll = hp.make_scroll_area(self)
        form = hp.make_form_layout(parent=panel)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addWidget(scroll, 1)
        form.addRow(
            hp.make_label(
                panel,
                "Uses accepted annotations whose object Training flag and class checkbox are enabled. Each object is an instance; classes are pooled into one foreground label. Annotate every instance in the sampled regions: unmarked pixels are background. Crops intersecting Ignore regions are skipped. At least two usable crops are needed for a separate validation split.",
                wrap=True,
            )
        )
        self.name = hp.make_line_edit(panel, text="Fine-tuned instance model")
        self.backend = hp.make_combobox(panel, ["cellpose", "micro-sam"])
        self.model = hp.make_combobox(panel)
        self.model.setMinimumContentsLength(8)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.download = hp.make_qta_btn(
            panel, "mdi6.download", tooltip="Download selected base model", func=self._download, size_preset="small"
        )
        self.help = hp.make_help_label(panel, "Choose a base model.")
        self.hint = hp.make_label(panel, "", wrap=True, enable_url=True)
        self.checkpoint = hp.make_line_edit(panel, placeholder="Optional local checkpoint")
        self.device = hp.make_combobox(panel, ["auto", "cpu", "cuda", "mps"])
        self.epochs = hp.make_int_spin_box(panel, minimum=1, maximum=10000, value=100)
        self.learning_rate = hp.make_line_edit(panel, text="0.00001")
        self.batch_size = hp.make_int_spin_box(panel, minimum=1, maximum=32, value=1)
        self.downsample = hp.make_int_spin_box(panel, minimum=1, maximum=256, value=1)
        self.patch_size = hp.make_combobox(panel, ["256", "512", "1024"])
        self.max_patches = hp.make_int_spin_box(panel, minimum=2, maximum=1024, value=128)
        self.freeze = hp.make_checkbox(panel, "Freeze SAM image encoder", default=True)
        self.freeze.setToolTip(
            "Train prompt/mask decoders while retaining the image encoder. Clear to fine-tune the full SAM model, which needs more memory."
        )
        form.addRow("Model name", self.name)
        form.addRow("Backend", self.backend)
        form.addRow("Base model", hp.make_h_layout(self.model, self.download, self.help, stretch_id=0))
        form.addRow(self.hint)
        form.addRow("Local checkpoint", self.checkpoint)
        form.addRow(hp.make_btn(panel, "Choose local checkpoint...", func=self._choose_checkpoint))
        for label, widget in (
            ("Device", self.device),
            ("Epochs", self.epochs),
            ("Learning rate", self.learning_rate),
            ("Batch size", self.batch_size),
            ("Working downsample", self.downsample),
            ("Source crop size (working px)", self.patch_size),
            ("Maximum crops", self.max_patches),
        ):
            form.addRow(label, widget)
        form.addRow(self.freeze)
        form.addRow(
            hp.make_label(
                panel,
                "The seed from Train controls the crop split and PyTorch initialization. About 20% of crops are reserved for validation. Cellpose retains its library's augmentation and epoch-shuffling policy. SAM trains its prompt/mask decoders; automatic prediction uses its mask generator. Auto uses CPU for SAM when only Apple MPS is available. New weights are saved separately under this project's models directory.",
                wrap=True,
            )
        )
        self.error = hp.make_label(self, "", wrap=True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.start = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.start.setText("Start fine-tuning")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.backend.currentTextChanged.connect(self._backend_changed)
        self.model.currentIndexChanged.connect(self._update)
        self.checkpoint.textChanged.connect(self._update)
        parent.jobs.completed.connect(self._completed)
        parent.jobs.failed.connect(self._completed)
        parent.jobs.busy_changed.connect(self._update)
        self._backend_changed()

    def _completed(self, _job: dict, _result: Any) -> None:
        self._refresh_models()

    def _backend_changed(self, *_args: Any) -> None:
        self.checkpoint.clear()
        self.window._ensure_model_catalogue(self.backend.currentText())
        self._refresh_models()

    def _refresh_models(self) -> None:
        current = self.model.currentData()
        with QSignalBlocker(self.model):
            self.model.clear()
            enabled = []
            for entry in self.window._catalogues.get(self.backend.currentText(), []):
                index = self.model.count()
                self.model.addItem(entry.get("display_name", entry["name"]), entry["name"])
                reason = entry.get("reason", "")
                self.model.setItemData(index, reason, Qt.ItemDataRole.ToolTipRole)
                self.model.model().item(index).setEnabled(not reason)
                if not reason:
                    enabled.append(index)
            index = self.model.findData(current)
            self.model.setCurrentIndex(index if index in enabled else enabled[0] if enabled else -1)
        self._update()

    def _entry(self) -> dict[str, Any]:
        return next(
            (
                entry
                for entry in self.window._catalogues.get(self.backend.currentText(), [])
                if entry["name"] == self.model.currentData()
            ),
            {},
        )

    def _update(self, *_args: Any) -> None:
        backend, entry = self.backend.currentText(), self._entry()
        self.freeze.setVisible(backend == "micro-sam")
        local = bool(self.checkpoint.text() and Path(self.checkpoint.text()).is_file())
        ready = (
            backend_available(backend)
            and bool(entry)
            and not entry.get("reason")
            and bool(local or entry.get("cached"))
        )
        explanation = escape(entry.get("reason") or "Download this model or choose a local checkpoint before starting.")
        if ready:
            directory = (
                Path(self.checkpoint.text() or entry["path"]).parent
                if self.checkpoint.text() or entry.get("path")
                else model_cache_directory(backend)
            )
            explanation = ("Local checkpoint in " if local else "Model cached in ") + hp.hyper(
                directory, value="model directory"
            )
        if not backend_available(backend):
            explanation = escape(f"Backend unavailable in the app's Python environment: {backend}.")
        self.hint.setText(
            explanation + "<br>" + model_requirements_html(backend, self.window._catalogues.get(backend, []))
        )
        self.help.setToolTip(
            pretrained_model_help(
                backend, entry.get("custom", {}).get("settings", {}).get("model", entry.get("name", ""))
            )
        )
        downloading = self.window.jobs.current and self.window.jobs.current["operation"] == "download_model"
        self.download.set_qta("mdi6.stop" if downloading else "mdi6.download")
        self.download.setToolTip("Cancel model download" if downloading else "Download selected base model")
        self.download.setEnabled(
            not self.window.jobs.cancelling
            if downloading
            else bool(entry and not entry.get("reason") and not entry.get("cached") and not self.window.jobs.busy)
        )
        self.start.setEnabled(ready and not self.window.jobs.busy)

    def _download(self) -> None:
        if self.window.jobs.current and self.window.jobs.current["operation"] == "download_model":
            self.window.jobs.cancel()
        else:
            self.window.jobs.submit(
                {
                    "operation": "download_model",
                    "backend": self.backend.currentText(),
                    "model": self.model.currentData(),
                }
            )

    def _choose_checkpoint(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Choose base checkpoint")
        if path:
            self.checkpoint.setText(path)

    def settings(self) -> FineTuneSettings:
        """Validate the explicit fine-tuning request without starting downloads or training."""
        entry = self._entry()
        if self.checkpoint.text() and not Path(self.checkpoint.text()).is_file():
            raise ValueError("Choose an existing local checkpoint.")
        custom = entry.get("custom", {})
        return FineTuneSettings(
            backend=self.backend.currentText(),
            model=custom.get("settings", {}).get("model", self.model.currentData() or ""),
            checkpoint=self.checkpoint.text() or (entry.get("path", "") if custom else ""),
            class_ids=self.class_ids,
            channels=self.window._channels(),
            downsample=self.downsample.value(),
            epochs=self.epochs.value(),
            learning_rate=float(self.learning_rate.text()),
            batch_size=self.batch_size.value(),
            patch_size=int(self.patch_size.currentText()),
            max_patches=self.max_patches.value(),
            seed=self.window.random_seed.value(),
            device=self.device.currentText(),
            freeze_encoder=self.freeze.isChecked(),
        )

    def _accept(self) -> None:
        try:
            self.settings()
        except ValueError as exc:
            self.error.setText(str(exc))
        else:
            self.accept()
