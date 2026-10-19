"""Qt polling for a persistent spawned annotation worker."""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from multiprocessing.process import BaseProcess
from queue import Empty
from uuid import uuid4

from qtpy.QtCore import QObject, QTimer, Signal  # type: ignore[attr-defined]

from image2image.utils.annotation.worker import worker_loop


class AnnotationJobs(QObject):
    """Run one compute job at a time with a bounded queue and explicit cancellation."""

    progress = Signal(str)
    completed = Signal(dict, object)
    failed = Signal(dict, str)
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._context = mp.get_context("spawn")
        self._input = self._context.Queue(maxsize=1)
        self._output = self._context.Queue(maxsize=128)
        self._cancel = self._context.Event()
        self._process: BaseProcess | None = None
        self._closed = False
        self.current: dict | None = None
        self.pending: deque[dict] = deque()
        self.timer = QTimer(self)
        self.timer.setInterval(75)
        self.timer.timeout.connect(self._poll)

    @property
    def busy(self) -> bool:
        """Whether a task is running or queued."""
        return self.current is not None or bool(self.pending)

    def submit(self, job: dict) -> None:
        """Queue an immutable operation; reject unbounded work submission."""
        if self._closed:
            raise RuntimeError("The annotation worker has been closed.")
        if len(self.pending) >= 64:
            raise ValueError("The queue is full (64 pending slides). Wait for existing jobs to finish.")
        self.pending.append({**job, "id": uuid4().hex})
        self._next()

    def _next(self) -> None:
        if self.current or not self.pending:
            return
        if self._process is None or not self._process.is_alive():
            self._process = self._context.Process(
                target=worker_loop, args=(self._input, self._output, self._cancel), daemon=True
            )
            self._process.start()
        self.current = self.pending.popleft()
        self._cancel.clear()
        self._input.put(self.current)
        self.busy_changed.emit(True)
        self.timer.start()

    def cancel(self) -> None:
        """Cancel the active operation and clear queued work."""
        self.pending.clear()
        self._cancel.set()
        self.progress.emit("Cancelling after the current bounded operation...")

    def _poll(self) -> None:
        while True:
            try:
                message = self._output.get_nowait()
            except Empty:
                break
            if self.current is None or message["id"] != self.current["id"]:
                continue
            if message["kind"] == "progress":
                self.progress.emit(message["message"])
                continue
            job, self.current = self.current, None
            if message["kind"] == "finished":
                self.completed.emit(job, message["result"])
            else:
                self.failed.emit(job, message["message"])
            self._next()
            self.busy_changed.emit(self.busy)
        if self.current and self._process and not self._process.is_alive():
            job, self.current = self.current, None
            self.pending.clear()
            self.failed.emit(
                job,
                "The worker stopped unexpectedly. Accepted annotations are safe; reopen the project to discard incomplete staging data.",
            )
            self.busy_changed.emit(False)
        if not self.busy:
            self.timer.stop()

    def close(self) -> None:
        """Stop the spawned process when its owning window closes."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self.pending.clear()
        self._cancel.set()
        if self._process and self._process.is_alive():
            if self.current is None:
                self._input.put(None)
            self._process.join(timeout=0.5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1)
        self.current = None
        self._input.close()
        self._output.close()
