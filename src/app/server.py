from typing import Optional
from PyQt6.QtCore import QThread, QMutex
from wtube.server_worker import DetectionReceiver, DetectionSender
from collections import deque


class DetectionLauncher:
    """
    Encapsulates all logic for launching and stopping DetectionWorker instances
    inside their own QThreads.
    """

    def __init__(self) -> None:
        self.receiver_thread: Optional[QThread] = None
        self.sender_thread: Optional[QThread] = None
        self.receiver: Optional[DetectionReceiver] = None
        self.sender: Optional[DetectionSender] = None

        self.result_queue = deque(maxlen=300)
        self.queue_mutex = QMutex()

    def start(self) -> None:
        """
        Starts the DetectionWorker threads if they are not already running.
        """
        print("Attempting to start DetectionLauncher")

        if (self.receiver_thread and self.receiver_thread.isRunning()) or \
           (self.sender_thread and self.sender_thread.isRunning()):
            print("DetectionLauncher is already running")
            return

        self.receiver_thread = QThread()
        self.receiver = DetectionReceiver(
            port=9001,
            result_queue=self.result_queue,
            queue_mutex=self.queue_mutex
        )
        self.receiver.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.receiver.process)

        self.sender_thread = QThread()
        self.sender = DetectionSender(
            port=9002,
            result_queue=self.result_queue,
            queue_mutex=self.queue_mutex
        )
        self.sender.moveToThread(self.sender_thread)
        self.sender_thread.started.connect(self.sender.process)

        self.receiver_thread.start()
        self.sender_thread.start()

    def stop(self) -> None:
        """
        Stops DetectionWorker threads and cleans up.
        """
        if self.receiver and self.receiver_thread and self.receiver_thread.isRunning():
            self.receiver.stop()
            self.receiver_thread.quit()
            self.receiver_thread.wait()
            self.receiver_thread = None
            self.receiver = None

        if self.sender and self.sender_thread and self.sender_thread.isRunning():
            self.sender.stop()
            self.sender_thread.quit()
            self.sender_thread.wait()
            self.sender_thread = None
            self.sender = None


if __name__ == '__main__':
    launcher = DetectionLauncher()
    launcher.start()
