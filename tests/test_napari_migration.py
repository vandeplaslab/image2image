"""Tests for napari 0.8 spatial calibration behavior."""

from __future__ import annotations

import typing as ty
import warnings
from types import SimpleNamespace

import numpy as np
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari.utils._units import get_unit_registry
from qtextraplot._napari.component_controls.qt_scalebar_controls import QtScaleBarControls
from qtextraplot._napari.image.components.viewer_model import Viewer
from qtpy.QtWidgets import QVBoxLayout, QWidget

from image2image.qt._dialog_base import BasePluginMixin
from image2image.qt._dialog_mixins import SingleViewerPluginMixin
from image2image.qt.dialog_register import ImageRegistrationPlugin
from image2image.utils.utilities import (
    MICROMETER_UNITS,
    PIXEL_UNITS,
    copy_layer_spatial_calibration,
    replace_shapes_layer_controls,
)


class ImageReader:
    """Minimal calibrated image reader for plotting tests."""

    reader_type = "image"
    resolution = 2.5
    scale = (2.5, 2.5)
    transform = np.eye(3)


class ShapesReader(ImageReader):
    """Minimal calibrated Shapes reader for plotting tests."""

    reader_type = "shapes"
    display_type = "shapes"

    @staticmethod
    def to_shapes_kwargs(**kwargs: ty.Any) -> dict[str, ty.Any]:
        """Return napari keyword arguments for one polygon."""
        data = np.array([[0, 0], [0, 1], [1, 1]])
        return {"data": [(data, "polygon")], **kwargs}


class PointsReader(ImageReader):
    """Minimal calibrated Points reader for plotting tests."""

    reader_type = "points"

    @staticmethod
    def to_points_kwargs(**kwargs: ty.Any) -> dict[str, ty.Any]:
        """Return napari keyword arguments for one point."""
        return {"data": np.array([[1, 1]]), **kwargs}


class ReaderWrapper:
    """Minimal reader wrapper used by the plotting mixin."""

    readers = (
        ("image", [np.zeros((4, 4))], ImageReader()),
        ("shapes", None, ShapesReader()),
        ("points", None, PointsReader()),
    )

    def channel_names(self) -> list[str]:
        """Return all layer names."""
        return [name for name, _, _ in self.readers]

    def channel_image_for_channel_names_iter(self, _names: list[str]) -> ty.Iterator[tuple[str, ty.Any, ImageReader]]:
        """Iterate over all test readers."""
        yield from self.readers

    @staticmethod
    def get_affine(_reader: ImageReader, _resolution: float) -> np.ndarray:
        """Return an identity physical transform."""
        return np.eye(3)


class RegistrationLayerHarness:
    """Minimal object for exercising registration layer properties."""

    def __init__(self) -> None:
        self.view_fixed = self._make_view()
        self.view_moving = self._make_view()
        self.fixed_point_size = SimpleNamespace(value=lambda: 5)
        self.moving_point_size = SimpleNamespace(value=lambda: 5)

    @staticmethod
    def _make_view() -> SimpleNamespace:
        """Return a model-only napari view with a fake visual mapping."""
        canvas = SimpleNamespace(layer_to_visual={})
        model = Viewer()

        class ViewerFacade:
            """Expose model methods while recording fake visuals."""

            def add_points(self, *args: ty.Any, **kwargs: ty.Any) -> Points:
                """Add a Points layer and register a placeholder visual."""
                layer = model.add_points(*args, **kwargs)
                canvas.layer_to_visual[layer] = SimpleNamespace()
                return layer

        return SimpleNamespace(viewer=ViewerFacade(), layers=model.layers, widget=SimpleNamespace(canvas=canvas))

    @staticmethod
    def on_run(*_args) -> None:
        """Receive layer data events during setup."""

    @staticmethod
    def on_predict(*_args) -> None:
        """Receive point-add events during setup."""


def _unit_names(layer: Layer) -> tuple[str, ...]:
    return tuple(map(str, layer.units))


def test_calibrated_reader_layers_use_micrometers() -> None:
    """Reader image, points, and shapes layers should share physical calibration."""
    viewer = Viewer()
    view = SimpleNamespace(viewer=viewer, layers=viewer.layers)
    model = SimpleNamespace(wrapper=ReaderWrapper())

    images, shapes, points = BasePluginMixin._plot_reader_layers(model, view, None, "view", scale=True)

    assert images and shapes and points
    for layer in [*images, *shapes, *points]:
        np.testing.assert_array_equal(layer.scale, (2.5, 2.5))
        assert _unit_names(layer) == ("micrometer", "micrometer")


def test_annotation_and_preview_layers_copy_reference_calibration() -> None:
    """New masks, labels, and previews should inherit their reference calibration."""
    reference = Image(np.zeros((8, 8)), scale=(0.75, 0.75), units=MICROMETER_UNITS)
    layers = [
        Shapes(),
        Points(np.empty((0, 2))),
        Labels(np.zeros((8, 8), dtype=np.uint8)),
        Image(np.zeros((8, 8))),
    ]

    for layer in layers:
        copy_layer_spatial_calibration(layer, reference)
        np.testing.assert_array_equal(layer.scale, reference.scale)
        assert layer.units == reference.units


def test_manual_reader_layers_use_pixels() -> None:
    """Manual registration reader layers should use unit pixel coordinates."""
    viewer = Viewer()
    view = SimpleNamespace(viewer=viewer, layers=viewer.layers)
    model = SimpleNamespace(wrapper=ReaderWrapper())

    images, shapes, points = BasePluginMixin._plot_reader_layers(model, view, None, "view", scale=False)

    assert images and shapes and points
    for layer in [*images, *shapes, *points]:
        np.testing.assert_array_equal(layer.scale, (1, 1))
        assert all(unit == get_unit_registry().pixel for unit in layer.units)


def test_resolution_edit_preserves_micrometer_units() -> None:
    """Updating reader scale should not return a calibrated layer to pixel units."""
    layer = Image(np.zeros((4, 4)), units=PIXEL_UNITS)
    reader = SimpleNamespace(key="reader", resolution=3.0, scale=(3.0, 3.0))
    wrapper = SimpleNamespace(
        channel_names_for_names=lambda _names: ["channel"],
        get_affine=lambda _reader, _resolution: np.eye(3),
    )
    plugin = SingleViewerPluginMixin.__new__(SingleViewerPluginMixin)
    plugin._image_widget = SimpleNamespace(
        model=SimpleNamespace(wrapper=wrapper, get_reader_for_key=lambda _key: reader)
    )
    plugin.view = SimpleNamespace(layers={"channel": layer})

    plugin.on_update_transform("reader")

    np.testing.assert_array_equal(layer.scale, (3.0, 3.0))
    assert _unit_names(layer) == ("micrometer", "micrometer")


def test_manual_registration_fiducials_use_pixels() -> None:
    """Manual registration fiducials should remain in pixel coordinates."""
    harness = RegistrationLayerHarness()

    layers = [
        ImageRegistrationPlugin.fixed_points_layer.fget(harness),
        ImageRegistrationPlugin.temporary_fixed_points_layer.fget(harness),
        ImageRegistrationPlugin.moving_points_layer.fget(harness),
        ImageRegistrationPlugin.temporary_moving_points_layer.fget(harness),
    ]
    for layer in layers:
        np.testing.assert_array_equal(layer.scale, (1, 1))
        assert all(unit == get_unit_registry().pixel for unit in layer.units)


def test_scalebar_controls_infer_layer_calibration_without_warning(qtbot) -> None:
    """qtextraplot controls should infer calibration without using ScaleBar.unit."""
    viewer = Viewer()
    viewer.add_image(np.zeros((4, 4)), scale=(2.5, 2.5), units=MICROMETER_UNITS)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        controls = QtScaleBarControls(viewer)
    qtbot.addWidget(controls)

    assert controls.units_combobox.currentData() == "um"
    assert controls.pixel_size.value() == 2.5


def test_shapes_controls_are_recreated_for_replacement_layer(qtbot) -> None:
    """Replacement controls should be composed around the new napari Shapes layer."""
    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    original = QtShapesControls(Shapes(name="original"))
    layout.addWidget(original)
    replacement_layer = Shapes(name="replacement")

    replacement = replace_shapes_layer_controls(original, replacement_layer)

    assert replacement is not original
    assert replacement.layer is replacement_layer
    assert layout.indexOf(replacement) >= 0


def test_mixed_resolution_layers_align_in_world_coordinates() -> None:
    """Equivalent positions at different resolutions should map to one physical point."""
    high_resolution = Image(np.zeros((16, 16)), scale=(0.5, 0.5), units=MICROMETER_UNITS)
    low_resolution = Image(np.zeros((4, 4)), scale=(2.0, 2.0), units=MICROMETER_UNITS)

    np.testing.assert_array_equal(high_resolution.data_to_world((8, 8)), low_resolution.data_to_world((2, 2)))
