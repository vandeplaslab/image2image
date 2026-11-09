"""Interactive localization-error trials for the Validate app: blind re-clicks and decoy calibration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from image2image.models.validation import LandmarkRepeat, ValidationDecoy, ValidationImage, ValidationLandmark
from image2image.utils.validation import sample_pixels, validation_contrast_limits, validation_image_pyramid

if TYPE_CHECKING:
    from image2image.qt.dialog_validate import ImageValidationWindow

HINT_PX = 15  # hint circle radius, in native pixels
DECOY_OFFSET_PX = 20  # maximum hidden decoy shift, in native pixels
DECOY_WINDOW_PX = 16  # half-size of the texture window used to choose decoy points
DECOY_CANDIDATES = 12  # random points tried per decoy; the most textured is used
HINT_LAYER = "Search hint"
DECOY_LAYERS = ("Decoy reference", "Decoy target", "Decoy point")


def _disk(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Uniform random XY offset within a disk."""
    angle, distance = rng.uniform(0, 2 * np.pi), radius * np.sqrt(rng.uniform())
    return distance * np.array([np.cos(angle), np.sin(angle)])


class LocalizationTrials:
    """Drive blind re-click and decoy trials in the two native views of the Validate window.

    While a trial runs, landmark markers, coordinate readouts and view linking are suspended so the user cannot
    simply copy earlier positions; they are restored when the trials finish or are stopped. Completed trials are
    kept when stopped early.
    """

    def __init__(self, window: ImageValidationWindow, seed: int | None = None) -> None:
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.kind: Literal["reclick", "decoy"] | None = None
        self._queue: list[Any] = []
        self._total = 0
        self._pending_fixed: tuple[float, float] | None = None
        self._repeats: dict[str, list[LandmarkRepeat]] = {}
        self._decoys: list[ValidationDecoy] = []
        self._current: ValidationDecoy | None = None
        self._saved: dict[str, Any] = {}
        self._pair_id = ""

    @property
    def active(self) -> bool:
        """Return whether a trial sequence is running."""
        return self.kind is not None

    # Starting and stopping
    def start_reclick(self, count: int) -> None:
        """Re-locate a random subset of enabled manual landmarks without showing their markers."""
        session = self.window.session
        if session.fixed is None or session.moving is None:
            self.window._set_status("Load both images before re-clicking landmarks.")
            return
        pool = [point for point in session.landmarks if point.enabled and point.source == "manual"]
        if not pool:
            self.window._set_status("Add manual landmarks before re-clicking them.")
            return
        order = self.rng.permutation(len(pool))[: min(count, len(pool))]
        self._begin("reclick", [pool[i] for i in order])
        self._next_reclick()

    def start_decoys(self, count: int) -> None:
        """Run decoy trials, alternating fixed and moving images, each paired with a hidden shifted copy."""
        session = self.window.session
        if session.fixed is None or session.moving is None or None in self.window._readers:
            self.window._set_status("Load both images before calibrating clicking.")
            return
        self._begin("decoy", [("fixed", "moving")[i % 2] for i in range(count)])
        self._next_decoy()

    def stop(self) -> None:
        """Finish the sequence, store completed trials and restore the views."""
        if not self.active:
            return
        from image2image.qt.dialog_validate import LandmarkCommand

        kind, window = self.kind, self.window
        self.kind = None
        self._remove_layers()
        self._restore()
        # Views can reload for another pair mid-trial, so store results in the pair the trials started on.
        pair = next((pair for pair in window.study.pairs if pair.id == self._pair_id), None)
        if pair is None:
            window._set_status("Stopped; the image pair was removed, so trials were discarded.")
        elif kind == "reclick" and self._repeats:
            before = pair.session.landmarks
            after = [
                point.model_copy(update={"repeats": [*point.repeats, *self._repeats[point.id]]})
                if point.id in self._repeats
                else point
                for point in before
            ]
            count = sum(len(items) for items in self._repeats.values())
            window.undo_stack.push(LandmarkCommand(window, pair.id, before, after, "Re-click landmarks"))
            window._set_status(f"Stored {count} blind re-click(s). Analyze to update localization error.")
        elif kind == "decoy" and self._decoys:
            pair.session.decoys = [*pair.session.decoys, *self._decoys]
            window._dirty = True
            if pair.id == window._pair_id:
                window.invalidate()
            else:
                window._drop_result(pair.id)
            window._set_status(f"Stored {len(self._decoys)} decoy trial(s). Analyze to update localization error.")
        else:
            window._set_status("Stopped; no trials were completed.")
        window._refresh_localization_label()

    def _begin(self, kind: Literal["reclick", "decoy"], queue: list[Any]) -> None:
        window = self.window
        window.cancel_pair()
        self.kind, self._queue, self._total = kind, list(queue), len(queue)
        self._pair_id = window._pair_id
        self._repeats, self._decoys, self._pending_fixed, self._current = {}, [], None, None
        self._saved = {
            "link": window.synchronize_zoom.isChecked(),
            "layers": [{layer.name: layer.visible for layer in view.viewer.layers} for view in window.views],
        }
        window.synchronize_zoom.setChecked(False)
        for layer in window.point_layers:
            layer.visible = False
        if kind == "decoy":
            for view in window.views:
                for layer in view.viewer.layers:
                    layer.visible = False
        for label in window.coordinate_labels:
            label.setVisible(False)
        window._cancel_shortcut.setEnabled(True)
        window.tabs.setCurrentIndex(1)

    def _restore(self) -> None:
        window = self.window
        for view, visibility in zip(window.views, self._saved.get("layers", []), strict=False):
            for layer in view.viewer.layers:
                if layer.name in visibility:
                    layer.visible = visibility[layer.name]
        for label in window.coordinate_labels:
            label.setVisible(True)
        window.synchronize_zoom.setChecked(self._saved.get("link", False))
        window._cancel_shortcut.setEnabled(False)

    def _remove_layers(self) -> None:
        for view in self.window.views:
            for name in (HINT_LAYER, *DECOY_LAYERS):
                if name in view.viewer.layers:
                    view.viewer.layers.remove(name)

    # Clicks
    def click(self, index: int, xy: tuple[float, float]) -> None:
        """Handle a left click in native view ``index`` at physical XY ``xy``."""
        if self.kind == "reclick":
            self._click_reclick(index, xy)
        elif self.kind == "decoy":
            self._click_decoy(index, xy)

    def _click_reclick(self, index: int, xy: tuple[float, float]) -> None:
        point: ValidationLandmark = self._queue[0]
        session = self.window.session
        if index == 0:
            if not session.fixed.geometry.contains(np.array(xy)):
                return
            self._pending_fixed = xy
            self._show_hint(1, np.array(point.moving), session.moving)
            self._status("Now click the same feature in the moving image.")
        elif self._pending_fixed is not None and session.moving.geometry.contains(np.array(xy)):
            self._repeats.setdefault(point.id, []).append(LandmarkRepeat(fixed=self._pending_fixed, moving=xy))
            self._queue.pop(0)
            self._next_reclick()
        else:
            self._status("Click the fixed image first.")

    def _click_decoy(self, index: int, xy: tuple[float, float]) -> None:
        if index != 1 or self._current is None:
            self._status("Click the matching feature in the right-hand (shifted) view.")
            return
        self._decoys.append(self._current.model_copy(update={"clicked": xy}))
        self._queue.pop(0)
        self._next_decoy()

    # Re-click trials
    def _next_reclick(self) -> None:
        if not self._queue:
            self.stop()
            return
        self._pending_fixed = None
        self._remove_layers()
        point = self._queue[0]
        self._show_hint(0, np.array(point.fixed), self.window.session.fixed)
        done = self._total - len(self._queue) + 1
        self._status(f"Re-click {done}/{self._total}: click the landmark feature inside the circle in the fixed image.")

    def _show_hint(self, index: int, xy: np.ndarray, image: ValidationImage) -> None:
        """Draw a search circle around a jittered center that still contains ``xy``; center the view there."""
        radius = HINT_PX * max(image.geometry.spacing)
        center = xy + _disk(self.rng, 0.6 * radius)
        viewer = self.window.views[index].viewer
        if HINT_LAYER in viewer.layers:
            viewer.layers.remove(HINT_LAYER)
        self.window._updating = True
        try:
            viewer.add_points(
                np.array([center[::-1]]),
                name=HINT_LAYER,
                size=2 * radius,
                face_color="transparent",
                border_color="magenta",
                border_width=0.05,
                units=("um", "um"),
            )
            viewer.camera.center = (*viewer.camera.center[:-2], *(center + _disk(self.rng, radius))[::-1])
        finally:
            self.window._updating = False
        if "Image" in viewer.layers:
            viewer.layers.selection.active = viewer.layers["Image"]

    # Decoy trials
    def _next_decoy(self) -> None:
        self._remove_layers()
        if not self._queue:
            self.stop()
            return
        role = self._queue[0]
        index = 0 if role == "fixed" else 1
        image = self.window.session.fixed if role == "fixed" else self.window.session.moving
        levels, rgb, scalar = self._levels(image, self.window._readers[index])
        spacing = np.asarray(image.geometry.spacing)
        point = self._textured_point(scalar, image)
        offset = _disk(self.rng, DECOY_OFFSET_PX * float(spacing.max()))
        self._current = ValidationDecoy(
            image=role, point=tuple(float(v) for v in point), offset=tuple(float(v) for v in offset)
        )
        limits = validation_contrast_limits(levels)
        data = levels if len(levels) > 1 else levels[0]
        window = self.window
        window._updating = True
        try:
            for view_index, name, shift in ((0, DECOY_LAYERS[0], np.zeros(2)), (1, DECOY_LAYERS[1], offset)):
                viewer = window.views[view_index].viewer
                layer = viewer.add_image(
                    data,
                    multiscale=len(levels) > 1,
                    rgb=rgb,
                    name=name,
                    scale=image.geometry.spacing[::-1],
                    translate=tuple((np.asarray(image.geometry.origin) + shift)[::-1]),
                    units=("um", "um"),
                    contrast_limits=limits,
                )
                viewer.layers.selection.active = layer
            reference = window.views[0].viewer
            reference.add_points(
                np.array([point[::-1]]),
                name=DECOY_LAYERS[2],
                size=6 * float(spacing.max()),
                canvas_size_limits=(8, 40),
                face_color="transparent",
                border_color="yellow",
                border_width=0.3,
                units=("um", "um"),
            )
            reference.layers.selection.active = reference.layers[DECOY_LAYERS[0]]
            zoom = reference.camera.zoom
            reference.camera.center = (*reference.camera.center[:-2], *point[::-1])
            target = window.views[1].viewer.camera
            target.zoom = zoom
            jitter = _disk(self.rng, DECOY_OFFSET_PX * float(spacing.max()))
            target.center = (*target.center[:-2], *(point + offset + jitter)[::-1])
        finally:
            window._updating = False
        done = self._total - len(self._queue) + 1
        self._status(
            f"Decoy {done}/{self._total} ({role} image): click the feature marked in yellow on the left "
            "in the right-hand view."
        )

    @staticmethod
    def _levels(image: ValidationImage, reader: Any) -> tuple[list[Any], bool, Any]:
        """Display pyramid of an image, whether it is RGB, and the scalar analysis channel at full resolution."""
        display = image.display_channel if image.display_channel is not None else image.channel
        rgb = display == "rgb" and image.is_rgb
        levels = validation_image_pyramid(reader, image.channel if rgb else int(display), rgb=rgb)
        return levels, rgb, reader.get_channel(image.channel, split_rgb=True)

    def _textured_point(self, array: Any, image: ValidationImage) -> np.ndarray:
        """Pick the most textured of several random interior points, so decoys have a findable feature."""
        geometry = image.geometry
        margin = np.minimum(DECOY_WINDOW_PX + DECOY_OFFSET_PX, (np.asarray(geometry.shape[::-1]) - 1) // 2)
        offsets = np.arange(-DECOY_WINDOW_PX, DECOY_WINDOW_PX + 1, 2)
        window = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)
        best, score = None, -1.0
        for _ in range(DECOY_CANDIDATES):
            pixel = self.rng.uniform(margin, np.asarray(geometry.shape[::-1]) - 1 - margin)
            values, valid = sample_pixels(array, geometry, geometry.to_physical(pixel + window))
            spread = float(np.std(values[valid])) if valid.any() else 0.0
            if spread > score:
                best, score = pixel, spread
        return geometry.to_physical(best)

    def _status(self, text: str) -> None:
        self.window._set_status(f"{text} Press Esc to stop and keep completed trials.")
