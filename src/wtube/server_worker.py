from __future__ import annotations

import socket
import struct
import json
import time
from typing import Tuple
from collections import deque

import numpy as np
import cv2
from numpy.typing import NDArray

from PyQt6.QtCore import QObject, pyqtSlot, QMutex

from utils.logging import log_frame, log_annotation, get_log, clear_log
from utils.config import load_config
from utils.registry import build, import_all_from_pipeline

from pathlib import Path


class DetectionWorker(QObject):
    def __init__(self, host='127.0.0.1', port=9001, result_queue: deque | None = None, queue_mutex: QMutex | None = None, parent=None):
        super().__init__(parent)

        self.host = host
        self.port = port
        self.server_sock = None
        self.client_conn = None
        self._running = False

        self.result_queue = result_queue
        self.queue_mutex = queue_mutex

    def start_server(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(1)
        print(f"[{self.__class__.__name__}] Listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        try:
            if self.client_conn:
                self.client_conn.shutdown(socket.SHUT_RDWR)
                self.client_conn.close()
        except Exception:
            pass
        try:
            if self.server_sock:
                self.server_sock.close()
        except Exception:
            pass

    def _read_exactly(self, conn: socket.socket, size: int) -> bytes:
        buf = b''
        while len(buf) < size:
            chunk = conn.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Connection closed by client")
            buf += chunk
        return buf


class DetectionReceiver(DetectionWorker):
    def __init__(self, host='127.0.0.1', port=9001, result_queue: deque | None = None, queue_mutex: QMutex | None = None, parent=None):
        super().__init__(host, port, result_queue, queue_mutex, parent)

        import_all_from_pipeline()

        config_path = Path(__file__).parent.parent / "wtube" / "configs" / "server_model.yml"
        cfg = load_config(str(config_path))

        self._pre = build("preprocess", cfg["preprocess"]["name"], **cfg["preprocess"]["args"])
        self._infer = build("inference", cfg["inference"]["name"], **cfg["inference"]["args"])
        self._post = build("postprocess", cfg["postprocess"]["name"], **cfg["postprocess"]["args"])

    @pyqtSlot()
    def process(self) -> None:
        self.start_server()
        self.client_conn, addr = self.server_sock.accept()
        print(f"[Receiver] Connected by {addr}")

        self._running = True
        conn = self.client_conn

        while self._running:
            try:
                frame, frame_num, frame_ts = self._receive_frame(conn)
            except Exception as e:
                print(f"[Receiver] Error receiving frame: {e}")
                break

            image_id = log_frame(frame, frame_num, frame_ts)

            try:
                batch = self._pre(frame)
                result = self._infer(batch)
                batch, bls = self._post(batch, result, frame)
            except Exception as e:
                print(f"[Receiver] Error during processing: {e}")
                if self.result_queue is not None and self.queue_mutex:
                    self.queue_mutex.lock()
                    try:
                        self.result_queue.append({"error": str(e)})
                    finally:
                        self.queue_mutex.unlock()
                continue

            log_annotation(image_id, bls, frame_ts)
            logs = get_log()
            logs_copy = {
                "images": list(logs["images"]),
                "annotations": list(logs["annotations"])
            }
            if self.result_queue is not None and self.queue_mutex:
                self.queue_mutex.lock()
                try:
                    self.result_queue.append({
                        "frame_num": frame_num,
                        "logs": logs_copy
                    })
                finally:
                    self.queue_mutex.unlock()
            clear_log()
            time.sleep(0.01)

        self.stop()

    def _receive_frame(self, conn: socket.socket) -> Tuple[NDArray[np.uint8], int, float]:
        header_len_bytes = self._read_exactly(conn, 4)
        header_len = struct.unpack("!I", header_len_bytes)[0]

        header_json = self._read_exactly(conn, header_len).decode("utf-8")
        header = json.loads(header_json)

        img_size = header["img_size"]
        frame_num = header["frame_num"]
        frame_ts = header["frame_ts"]

        img_bytes = self._read_exactly(conn, img_size)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image")

        return frame, frame_num, frame_ts


class DetectionSender(DetectionWorker):
    def __init__(self, host='127.0.0.1', port=9002, result_queue: deque | None = None, queue_mutex: QMutex | None = None, parent=None):
        super().__init__(host, port, result_queue, queue_mutex, parent)

    @pyqtSlot()
    def process(self) -> None:
        self.start_server()
        self.client_conn, addr = self.server_sock.accept()
        print(f"[Sender Server] Connected by {addr}")

        self._running = True
        conn = self.client_conn

        while self._running:
            try:
                result = None
                if self.result_queue is not None and self.queue_mutex:
                    self.queue_mutex.lock()
                    try:
                        if len(self.result_queue) > 0:
                            result = self.result_queue.popleft()
                    finally:
                        self.queue_mutex.unlock()

                if result is None:
                    time.sleep(0.01)
                    continue

                self._send_response(conn, result)
            except Exception as e:
                print(f"[Sender] Error sending response: {e}")
                break

        self.stop()

    def _send_response(self, conn: socket.socket, response_dict: dict) -> None:
        resp_json = json.dumps(response_dict).encode("utf-8")
        conn.settimeout(10.0)
        conn.sendall(struct.pack("!I", len(resp_json)))
        conn.sendall(resp_json)
        conn.settimeout(None)
