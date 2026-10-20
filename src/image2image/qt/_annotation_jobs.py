"""Qt polling for a persistent spawned annotation worker."""

from __future__ import annotations

import multiprocessing as mp
from collections import deque
from multiprocessing.process import BaseProcess
from queue import Empty
from time import monotonic
from uuid import uuid4

from qtpy.QtCore import QObject, QTimer, Signal  # type: ignore[attr-defined]

from image2image.utils.annotation.model_cache import cleanup_model_download
from image2image.utils.annotation.worker import worker_loop


class AnnotationJobs(QObject):
    """Run one compute job at a time with a bounded queue and explicit cancellation."""

    progress = Signal(str)
    details = Signal(dict)
    completed = Signal(dict, object)
    failed = Signal(dict, str)
    busy_changed = Signal(bool)
    queue_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._context = mp.get_context("spawn")
        self._input = self._context.Queue(maxsize=1)
        self._output = self._context.Queue(maxsize=128)
        self._cancel = self._context.Event()
        self._process: BaseProcess | None = None
        self._closed = False
        self._started = 0.0
        self._phase = ""
        self._phase_samples: deque[tuple[float, int]] = deque(maxlen=8)
        self._last_progress: dict = {}
        self._reported = 0.0
        self.cancelling = False
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
        self.queue_changed.emit()

    def _next(self) -> None:
        if self.current or not self.pending:
            return
        if self._process is None or not self._process.is_alive():
            self._process = self._context.Process(
                target=worker_loop, args=(self._input, self._output, self._cancel), daemon=True
            )
            self._process.start()
        self.current = self.pending.popleft()
        self.cancelling = False
        self._started = monotonic()
        self._phase = ""
        self._phase_samples.clear()
        self._cancel.clear()
        self._input.put(self.current)
        self.busy_changed.emit(True)
        self._report_progress(
            {
                "phase": "Preparing",
                "completed": 0,
                "total": None,
                "message": f"Preparing {self.current['operation'].replace('_', ' ')}",
            }
        )
        self.timer.start()

    def _report_progress(self, message: dict) -> None:
        now = monotonic()
        self._last_progress, self._reported = message, now
        phase = message.get("phase", "Working")
        completed, total = message.get("completed", 0), message.get("total")
        changed_phase = phase != self._phase
        if changed_phase or (self._phase_samples and completed < self._phase_samples[-1][1]):
            self._phase, self._phase_samples = phase, deque(maxlen=8)
        if not self._phase_samples or completed > self._phase_samples[-1][1]:
            self._phase_samples.append((now, completed))
        eta = None
        if len(self._phase_samples) >= 2 and total is not None and total > completed:
            start, count = self._phase_samples[0]
            measured_at, measured_count = self._phase_samples[-1]
            elapsed, done = measured_at - start, measured_count - count
            if elapsed > 0 and done > 0:
                eta = max(0, elapsed / done * (total - completed) - (now - measured_at))
        current = self.current or {}
        self.details.emit(
            {
                **message,
                "elapsed": max(0, now - self._started),
                "eta": eta,
                "position": current.get("queue_index", 1),
                "queue_total": current.get("queue_total", 1),
                "queued": len(self.pending),
            }
        )
        self.progress.emit(message["message"])
        if changed_phase:
            self.queue_changed.emit()

    def cancel(self) -> None:
        """Cancel the active operation and clear queued work."""
        pending = list(self.pending)
        self.pending.clear()
        self.cancelling = self.current is not None
        self._cancel.set()
        for job in pending:
            self.failed.emit(job, "Operation cancelled before starting.")
        self.progress.emit("Cancelling after the current bounded operation...")
        self.queue_changed.emit()
        if self.current and self.current["operation"] == "download_model":
            job_id = self.current["id"]
            QTimer.singleShot(300, lambda: self._cancel_download(job_id))

    def _cancel_download(self, job_id: str) -> None:
        if not self.current or self.current["id"] != job_id:
            return
        # Downloads have no project transaction. Stop blocked network reads and replace
        # the process queues, which may have been interrupted during a progress write.
        job, self.current = self.current, None
        if self._process:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=1)
        self._process = None
        for queue in (self._input, self._output):
            queue.cancel_join_thread()
            queue.close()
        self._input, self._output = self._context.Queue(maxsize=1), self._context.Queue(maxsize=128)
        self._cancel = self._context.Event()
        message = "Model download cancelled. Incomplete files were removed; you can retry."
        try:
            cleanup_model_download(job["backend"], job_id)
        except OSError as exc:
            message = f"Model download cancelled, but incomplete files could not be removed: {exc}"
        self.cancelling = False
        self.failed.emit(job, message)
        self.busy_changed.emit(False)
        self.queue_changed.emit()
        self.timer.stop()

    def _poll(self) -> None:
        while True:
            try:
                message = self._output.get_nowait()
            except Empty:
                break
            if self.current is None or message["id"] != self.current["id"]:
                continue
            if message["kind"] == "progress":
                if not self.cancelling:
                    self._report_progress(message)
                continue
            job, self.current = self.current, None
            self.cancelling = False
            if message["kind"] == "finished":
                self.completed.emit(job, message["result"])
            else:
                self.failed.emit(job, message["message"])
            self._next()
            self.busy_changed.emit(self.busy)
            self.queue_changed.emit()
        if self.current and self._process and not self._process.is_alive():
            job, self.current = self.current, None
            self.pending.clear()
            self.failed.emit(
                job,
                "The worker stopped unexpectedly. Accepted annotations are safe; reopen the project to discard incomplete staging data.",
            )
            self.busy_changed.emit(False)
            self.queue_changed.emit()
        if self.current and not self.cancelling and monotonic() - self._reported >= 1:
            self._report_progress(self._last_progress)
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
