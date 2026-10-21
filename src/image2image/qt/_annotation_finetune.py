"""Explicit model and optimization choices for annotation-based instance fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import qtextra.helpers as hp
from qtpy.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QFormLayout

from image2image.models.annotation import FineTuneSettings
from image2image.qt._annotation_model import ModelSelector
from image2image.utils.annotation.ml import backend_available

if TYPE_CHECKING:
    from image2image.qt.dialog_annotate import ImageAnnotationWindow


class FineTuneDialog(QDialog):
    """Choose an available base checkpoint and inspect instance-training assumptions before starting."""

    def __init__(self, parent: ImageAnnotationWindow, class_ids: tuple[str, ...]) -> None:
        super().__init__(parent)
        self.window, self.class_ids = parent, class_ids
        self._backend_models: dict[str, str] = {}
        self._active_backend = ""
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
        self.model_selector = ModelSelector(panel, parent.jobs, parent.on_pressed_link)
        self.model = self.model_selector.model
        self.download, self.help = self.model_selector.download, self.model_selector.help
        self.hint = self.model_selector.status
        self.download.clicked.connect(self._download)
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
        form.addRow(hp.make_label(panel, "Base model"))
        form.addRow(self.model_selector)
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
        parent.jobs.queue_changed.connect(self._update)
        self._backend_changed()

    def _completed(self, _job: dict, _result: Any) -> None:
        self._refresh_models()

    def _backend_changed(self, *_args: Any) -> None:
        if self._active_backend:
            self._backend_models[self._active_backend] = self.model.currentData() or ""
        self._active_backend = self.backend.currentText()
        self.checkpoint.clear()
        self.window._ensure_model_catalogue(self.backend.currentText())
        self._refresh_models(self._backend_models.get(self._active_backend, ""))

    def _refresh_models(self, preferred: str | None = None) -> None:
        backend = self.backend.currentText()
        self.model_selector.set_catalogue(
            backend,
            self.window._catalogues.get(backend, []),
            self.model.currentData() if preferred is None else preferred,
        )
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
        self.model_selector.update_state(
            backend,
            entry,
            self.window._catalogues.get(backend, []),
            available=backend_available(backend),
            checkpoint=self.checkpoint.text(),
        )
        self.start.setEnabled(bool(entry) and self.model_selector.ready and not self.window.jobs.busy)

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
