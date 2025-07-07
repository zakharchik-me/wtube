from typing import Optional
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, pyqtSlot, QTimer

from app.viewer import ImageViewer
from client.client_worker import FrameWorker, NetworkWorker, NetworkSender, NetworkListener
from utils.config import load_config
from utils.registry import build, import_all_from_pipeline


class ClientLauncher(QObject):
    frame_changed = pyqtSignal(int)
    frame_received = pyqtSignal(int)
    frame_buffered = pyqtSignal(int)
    log_buffered = pyqtSignal(int)

    def __init__(self, viewer: Optional[ImageViewer] = None) -> None:
        super().__init__()
        self.viewer = viewer
        self.sender_thread: Optional[QThread] = None
        self.listener_thread: Optional[QThread] = None
        self.frame_thread: Optional[QThread] = None
        self.listen_thread: Optional[QThread] = None

        self.network_sender = None
        self.network_listener = None
        self.frame_worker = None
        self.source = None

    def start(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        import_all_from_pipeline()

        if config_dict is not None:
            cfg = config_dict
        elif config_path is not None:
            cfg = load_config(config_path)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.source = build("source", cfg["source"]["name"], **cfg["source"]["args"])

        self.frame_worker = FrameWorker(self.source)

        self.network_sender = NetworkSender(self.source)
        self.network_listener = NetworkListener()

        self.sender_thread = QThread()
        self.listener_thread = QThread()
        self.frame_thread = QThread()

        self.network_sender.moveToThread(self.sender_thread)
        self.network_listener.moveToThread(self.listener_thread)
        self.frame_worker.moveToThread(self.frame_thread)

        self.network_listener.log_received.connect(self.frame_worker.handle_log)
        # self.network_listener.log_received.connect(lambda log, num: print("Got log in test slot:", log, num))
        self.network_sender.frame_sent.connect(self.frame_worker.handle_result)

        self.frame_worker.frame_ready.connect(self.viewer.update_image)
        self.frame_worker.frame_index_changed.connect(self.frame_changed)
        self.frame_worker.frame_buffered.connect(self.frame_buffered)
        self.frame_worker.log_buffered.connect(self.log_buffered)

        self.sender_thread.started.connect(self.network_sender.send)
        self.listener_thread.started.connect(lambda: QTimer.singleShot(0, self.network_listener.listen))
        self.frame_thread.started.connect(lambda: None)

        self.sender_thread.start()
        self.listener_thread.start()
        self.frame_thread.start()

    def stop(self):
        if self.network_sender:
            self.network_sender.stop()
        if self.network_listener:
            self.network_listener.stop()
        if self.frame_worker:
            self.frame_worker.stop()
        if self.sender_thread:
            self.sender_thread.quit()
            self.sender_thread.wait()
        if self.listener_thread:
            self.listener_thread.quit()
            self.listener_thread.wait()
        if self.frame_worker:
            self.frame_worker.stop()
        if self.frame_thread:
            self.frame_thread.quit()
            self.frame_thread.wait()

    def pause(self):
        if self.frame_worker:
            self.frame_worker.pause()

    def play(self):
        if self.frame_worker:
            self.frame_worker.resume()
