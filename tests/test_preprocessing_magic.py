"""Tests for Magic preprocessing translation controls."""

from __future__ import annotations

import numpy as np
from image2image_reg.models import Modality, Preprocessing

from image2image.qt._wsi._list import QtModalityList
from image2image.qt._wsi._preprocessing import PreprocessingDialog


def _make_modality(name: str = "source") -> Modality:
    """Create a minimal modality for preprocessing dialog tests."""
    preprocessing = Preprocessing.basic(channel_indices=[0], channel_names=["channel"])
    return Modality(
        name=name,
        path=np.zeros((16, 16), dtype=np.uint8),
        channel_names=["channel"],
        preprocessing=preprocessing,
    )


def test_magic_result_replaces_translation_and_previews(qtbot) -> None:
    """Magic results update the working translation and request a spatial preview."""
    modality = _make_modality()
    dialog = PreprocessingDialog(modality)
    qtbot.addWidget(dialog)
    previewed = []
    dialog.evt_preview_transform_preprocessing.connect(lambda *_: previewed.append(True))

    dialog._on_magic_finished((126.4, -17.6))

    assert dialog.preprocessing.translate_x == 126
    assert dialog.preprocessing.translate_y == -18
    qtbot.waitUntil(lambda: bool(previewed), timeout=1_000)
    assert previewed == [True]


def test_magic_cancelled_result_is_ignored(qtbot) -> None:
    """A late worker result cannot overwrite translation after cancellation."""
    dialog = PreprocessingDialog(_make_modality())
    qtbot.addWidget(dialog)
    dialog.preprocessing.translate_x = 10
    dialog.preprocessing.translate_y = 20
    dialog._magic_cancelled = True

    dialog._on_magic_finished((126.4, -17.6))

    assert dialog.preprocessing.translate_x == 10
    assert dialog.preprocessing.translate_y == 20


def test_magic_targets_exclude_the_source_modality() -> None:
    """Target selection never presents the image being translated."""
    source = _make_modality("source")
    target = _make_modality("target")

    class FakeModalityList:
        def model_iter(self):
            return [source, target]

    assert QtModalityList.get_magic_targets(FakeModalityList(), source) == [target]


def test_valis_keeps_only_translation_controls_unlocked(qtbot) -> None:
    """Valis allows Magic translation refinement without unlocking other spatial settings."""
    dialog = PreprocessingDialog(_make_modality(), valis=True)
    qtbot.addWidget(dialog)

    assert dialog.translate_x.isEnabled()
    assert dialog.translate_y.isEnabled()
    assert dialog.magic_btn.isEnabled()
    assert not dialog.rotate_spin.isEnabled()
    assert not dialog.downsample_spin.isEnabled()
