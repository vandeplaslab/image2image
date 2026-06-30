"""Tests for dataset loading dialog behavior."""

from __future__ import annotations

import typing as ty

from image2image.qt._dialogs import _dataset


class FakeSignal:
    """Minimal signal that records emitted arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[ty.Any, ...]] = []

    def emit(self, *args: ty.Any) -> None:
        """Record emitted arguments."""
        self.calls.append(args)


class FakeWrapper:
    """Minimal wrapper that returns channel names for requested keys."""

    def __init__(self, channels: list[str]) -> None:
        self.channels = channels
        self.requested_keys: list[list[str]] = []

    def channel_names_for_names(self, keys: list[str]) -> list[str]:
        """Return configured channel names."""
        self.requested_keys.append(keys)
        return self.channels


class FakeModel:
    """Minimal data model for loaded dataset tests."""

    def __init__(self, channels: list[str], just_added_keys: list[str] | None = None) -> None:
        self.wrapper = FakeWrapper(channels)
        self.just_added_keys = just_added_keys or ["image"]
        self.removed_keys: list[list[str]] = []

    def remove_keys(self, keys: list[str]) -> None:
        """Record removed keys."""
        self.removed_keys.append(keys)


class FakeDatasetDialog:
    """Minimal dialog object with attributes used by DatasetDialog methods."""

    def __init__(self, allow_channels: bool = True, is_fixed: bool = False) -> None:
        self.allow_channels = allow_channels
        self.is_fixed = is_fixed
        self.evt_loaded = FakeSignal()
        self.evt_loaded_keys = FakeSignal()
        self.populate_count = 0

    def on_populate_table(self) -> None:
        """Record table refreshes."""
        self.populate_count += 1


class FakeSelectChannelsToLoadDialog:
    """Minimal channel selection dialog."""

    created_count = 0

    def __init__(self, _parent: object, _model: FakeModel) -> None:
        type(self).created_count += 1
        self.channels = ["selected | image"]

    def show_in_center_of_screen(self) -> None:
        """No-op for tests."""

    def raise_(self) -> None:
        """No-op for tests."""

    def exec_(self) -> bool:
        """Accept the dialog."""
        return True


def test_loaded_dataset_auto_accepts_single_channel(monkeypatch) -> None:
    """Single-channel loads should skip the channel selection dialog."""
    model = FakeModel(["tic | image"])
    dialog = FakeDatasetDialog()

    def fail_dialog(*_args: ty.Any, **_kwargs: ty.Any) -> None:
        msg = "SelectChannelsToLoadDialog should not be created for one channel."
        raise AssertionError(msg)

    monkeypatch.setattr(_dataset, "SelectChannelsToLoadDialog", fail_dialog)

    _dataset.DatasetDialog._on_loaded_dataset(dialog, model)

    assert dialog.evt_loaded.calls == [(model, ["tic | image"])]
    assert dialog.evt_loaded_keys.calls == [(["image"],)]
    assert model.removed_keys == []
    assert dialog.populate_count == 1


def test_loaded_dataset_opens_dialog_for_multiple_channels(monkeypatch) -> None:
    """Multi-channel loads should keep the existing selection dialog."""
    model = FakeModel(["red | image", "green | image"])
    dialog = FakeDatasetDialog()
    FakeSelectChannelsToLoadDialog.created_count = 0
    monkeypatch.setattr(_dataset, "SelectChannelsToLoadDialog", FakeSelectChannelsToLoadDialog)

    _dataset.DatasetDialog._on_loaded_dataset(dialog, model)

    assert FakeSelectChannelsToLoadDialog.created_count == 1
    assert dialog.evt_loaded.calls == [(model, ["selected | image"])]
    assert model.removed_keys == []


def test_loaded_dataset_with_preselection_auto_accepts_single_channel(monkeypatch) -> None:
    """Preselected single-channel loads should skip the channel selection dialog."""
    model = FakeModel(["tic | image"])
    dialog = FakeDatasetDialog()

    def fail_dialog(*_args: ty.Any, **_kwargs: ty.Any) -> None:
        msg = "SelectChannelsToLoadDialog should not be created for one channel."
        raise AssertionError(msg)

    monkeypatch.setattr(_dataset, "SelectChannelsToLoadDialog", fail_dialog)

    _dataset.DatasetDialog._on_loaded_dataset_with_preselection(dialog, model)

    assert dialog.evt_loaded.calls == [(model, ["tic | image"])]
    assert model.removed_keys == []
    assert dialog.populate_count == 1
