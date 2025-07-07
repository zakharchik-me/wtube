import pytest

from workers.detection_worker import DetectionWorker


def test_worker_has_signal_and_slots() -> None:
    """Verify that DetectionWorker has the expected signal and slot."""
    worker = DetectionWorker()
    assert hasattr(worker, 'boxes_ready')
    assert callable(getattr(worker, 'process'))