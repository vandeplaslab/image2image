"""Shared model selection and download presentation for annotation workflows."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any

import qtextra.helpers as hp
from qtpy.QtCore import QSignalBlocker, Qt
from qtpy.QtWidgets import QComboBox, QSizePolicy, QWidget

from image2image.qt._annotation_jobs import AnnotationJobs
from image2image.utils.annotation.model_cache import model_cache_directory, pretrained_model_help


def model_requirements_html(backend: str, entries: list[dict[str, Any]]) -> str:
    """Explain unavailable architectures beside a model selector."""
    families: dict[str, list[str]] = {}
    for entry in entries:
        if entry.get("reason"):
            families.setdefault(entry["reason"], []).append(entry.get("display_name", entry["name"]))
    if not families:
        return ""
    message = "<br>".join(f"<b>{escape(', '.join(names))}</b>: {escape(reason)}" for reason, names in families.items())
    url = (
        "https://computational-cell-analytics.github.io/micro-sam/#installation"
        if backend == "micro-sam"
        else "https://cellpose.readthedocs.io/en/latest/_modules/cellpose/models.html"
    )
    return message + "<br>" + hp.hyper(url, value="Backend installation instructions")


class ModelSelector(QWidget):
    """Present model choices, cache state, requirements, and download controls consistently."""

    def __init__(self, parent: QWidget, jobs: AnnotationJobs, on_link: Callable[[str], None]) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.ready = False
        self.model = hp.make_combobox(self)
        self.model.setPlaceholderText("No models available")
        self.model.setMinimumContentsLength(8)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.download = hp.make_qta_btn(self, "download", tooltip="Download selected model", size_preset="small")
        self.help = hp.make_help_label(self, "Select a model to see its purpose and requirements.")
        self.status = hp.make_label(self, "", wrap=True, enable_url=True, func_activated=on_link)
        self.requirements = hp.make_label(self, "", wrap=True, enable_url=True)
        layout = hp.make_form_layout(parent=self, margin=0, spacing=2)
        layout.addRow(hp.make_h_layout(self.model, self.download, self.help, stretch_id=0, spacing=2))
        layout.addRow(self.status)
        layout.addRow(self.requirements)

    def set_entries(self, backend: str, entries: list[tuple[str, str, str]], preferred: str = "") -> None:
        """Populate labelled model IDs while retaining an available preferred selection."""
        with QSignalBlocker(self.model):
            self.model.clear()
            enabled = []
            for label, value, reason in entries:
                index = self.model.count()
                self.model.addItem(label, value)
                help_text = pretrained_model_help(backend, value) if backend != "forest" else value
                help_text = escape(f"{help_text}\n{reason}" if reason else help_text).replace("\n", "<br>")
                self.model.setItemData(index, f"<qt>{help_text}</qt>", Qt.ItemDataRole.ToolTipRole)
                self.model.model().item(index).setEnabled(not reason)
                if not reason:
                    enabled.append(index)
            index = self.model.findData(preferred)
            self.model.setCurrentIndex(index if index in enabled else enabled[0] if enabled else -1)

    def set_catalogue(self, backend: str, entries: list[dict[str, Any]], preferred: str = "") -> None:
        """Display catalogue names with their current download state."""
        self.set_entries(
            backend,
            [
                (
                    f"{entry.get('display_name', entry['name'])} ({'cached' if entry['cached'] else 'not downloaded'})",
                    entry["name"],
                    entry.get("reason", ""),
                )
                for entry in entries
            ],
            preferred,
        )

    def update_state(
        self,
        backend: str,
        entry: dict[str, Any] | None,
        entries: list[dict[str, Any]],
        *,
        available: bool,
        checkpoint: str = "",
    ) -> None:
        """Refresh readiness, cache links, help, and download cancellation state."""
        forest = backend == "forest"
        local = bool(checkpoint and Path(checkpoint).is_file())
        reason = entry.get("reason", "") if entry else ""
        self.ready = available and not reason and bool(local or (entry and entry.get("cached")))
        downloading = self.jobs.current and self.jobs.current.get("operation") == "download_model"
        self.download.setVisible(not forest)
        self.download.set_qta("mdi6.stop" if downloading else "download")
        self.download.setToolTip(
            f"Cancel download: {self.jobs.current['model']}" if downloading else "Download selected model"
        )
        self.download.setEnabled(
            not self.jobs.cancelling
            if downloading
            else bool(available and entry and not reason and not entry.get("cached") and not self.jobs.busy)
        )
        self.status.setVisible(not forest)
        requirements = model_requirements_html(backend, entries) if not forest else ""
        self.requirements.setText(requirements)
        self.requirements.setVisible(bool(requirements))
        if forest:
            self.status.setText("")
            return
        directory = model_cache_directory(backend)
        if not available:
            message = f"Backend unavailable in the app's Python environment: {escape(backend)}."
        elif reason:
            message = escape(reason)
        elif entry and entry.get("custom") and (not local or Path(checkpoint) == Path(entry["path"])):
            directory = Path(entry["path"]).parent
            message = "Project model in " + hp.hyper(directory, value="model directory") + "."
        elif local:
            directory = Path(checkpoint).parent
            message = "Using local checkpoint in " + hp.hyper(directory, value="model directory") + "."
        elif self.ready:
            message = "Model cached in " + hp.hyper(directory, value=escape(backend + " model directory")) + "."
        else:
            message = "Not downloaded. Choose a model, then Download selected model."
        self.status.setText(message)
        self.status.setToolTip(str(directory))
        model = entry.get("custom", {}).get("settings", {}).get("model", entry.get("name", "")) if entry else ""
        help_text = pretrained_model_help(backend, model)
        self.help.setToolTip(
            "<qt>" + escape(f"{help_text}\n{reason}" if reason else help_text).replace("\n", "<br>") + "</qt>"
        )
