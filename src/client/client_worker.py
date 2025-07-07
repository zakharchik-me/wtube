from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from .buffer import RollingBuffer
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import os
import cv2
from .buffer import FolderBuffer
from utils.video_saver import VideoSaver
import time


import socket
import struct
import json
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal


class NetworkWorker(QObject):
    def __init__(self, host='127.0.0.1', port=9002):
        super().__init__()
        self.host = host
        self.port = port
        self._running = False
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.sock.setblocking(True)

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()
            self.sock = None


class NetworkSender(NetworkWorker):
    frame_sent = pyqtSignal(object, int, float)

    def __init__(self, source, host='127.0.0.1', port=9001):
        super().__init__(host, port)
        self.source = source
        self.host = host
        self.port = port

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.sock.setblocking(True)

    @pyqtSlot()
    def send(self):
        self.connect()
        self._running = True
        while self._running:
            try:
                frame, frame_num, frame_ts = next(self.source)
                start_time = time.time()
                self.send_frame(frame, frame_num, frame_ts)
                elapsed_time = time.time() - start_time
                self.frame_sent.emit(frame, frame_num, frame_ts)
                print(f"[SENDER] Elapsed: {elapsed_time*1000:.2f} ms")
            except StopIteration:
                break
            except Exception as e:
                print(f"[SENDER] Error: {e}")
                break
            time.sleep(0.001)

    def send_frame(self, frame, frame_num, frame_ts):
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        header = json.dumps({
            "frame_num": frame_num,
            "frame_ts": frame_ts,
            "img_size": len(img_bytes)
        }).encode("utf-8")

        header_len = struct.pack("!I", len(header))
        self.sock.settimeout(10.0)
        self.sock.sendall(header_len + header + img_bytes)
        self.sock.settimeout(None)


class NetworkListener(NetworkWorker):
    response_received = pyqtSignal(dict)
    log_received = pyqtSignal(dict, int)


    def connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.sock.setblocking(True)

    @pyqtSlot()
    def listen(self):
        self._running = True
        while self._running:
            try:
                if self.sock is None:
                    try:
                        self.connect()
                    except Exception as e:
                        time.sleep(1)
                        continue

                response = self.receive_response()
                self.response_received.emit(response)
                frame_num = response.get("frame_num", 0)
                self.log_received.emit(response, frame_num)

            except ConnectionError as e:
                print(f"[LISTENER] Connection lost: {e}")
                if self.sock:
                    self.sock.close()
                    self.sock = None
                time.sleep(1)

            except Exception as e:
                print(f"[LISTENER] Unexpected error: {e}")
                time.sleep(0.1)

    def receive_response(self):
        resp_len_bytes = self._read_exactly(4)
        resp_len = struct.unpack("!I", resp_len_bytes)[0]
        resp_json = self._read_exactly(resp_len).decode("utf-8")
        return json.loads(resp_json)

    def _read_exactly(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf


class FrameWorker(QObject):
    frame_ready = pyqtSignal(object, object, object, object)
    frame_index_changed = pyqtSignal(int)
    frame_buffered = pyqtSignal(int)
    log_buffered = pyqtSignal(int)

    def __init__(self, source, max_buffer_size=300, save_dir='logs/'):
        super().__init__()
        self.save_dir = os.path.join(save_dir, os.path.basename(source.get_src_name()))
        print(self.save_dir)
        video_saver_instance = VideoSaver(path=self.save_dir, max_size=max_buffer_size)
        self._disk = video_saver_instance
        self._buffer = RollingBuffer(max_size=max_buffer_size, video_saver=video_saver_instance)

        self._paused = False
        self._last_frame_index = 1
        self._next_frame_to_show = 1
        self._timer = QTimer()
        self._timer.timeout.connect(self.next_frame)
        self._interval = 1000 / source.get_fps()
        print(self._interval)

    @pyqtSlot(dict, int)
    def handle_log(self, log, frame_num):
        self.log_buffered.emit(frame_num)
        self._buffer.add_ann(log, frame_num)

        if self._paused:
            return

        if frame_num == self._next_frame_to_show:
            self._show_frame(frame_num)
            self._next_frame_to_show += 1

    def pause(self):
        self._paused = True
        self._timer.stop()

    def resume(self):
        self._paused = False
        self._timer.start(self._interval)

    def stop(self):
        self._paused = True
        self._timer.stop()
        self._buffer.flush()
        self._disk.close()

    @pyqtSlot(int)
    def seek(self, frame_index: int):
        self._timer.stop()
        self._paused = True

        self._next_frame_to_show = frame_index
        self._show_frame(frame_index)
        self._next_frame_to_show = frame_index + 1

    def get_max_ready(self):
        return self._buffer.get_all_indices()[-1]

    @pyqtSlot(object, int)
    def handle_result(self, frame, frame_num: int):
        self.frame_buffered.emit(frame_num)
        self._buffer.add_image(frame, frame_num)

        if self._paused:
            return

    def next_frame(self):
        next_index = self._next_frame_to_show
        all_indices = self._buffer.get_all_indices()

        if all_indices and next_index <= all_indices[-1]:
            self._show_frame(next_index)
            self._next_frame_to_show += 1
        else:
            pass

    def _show_frame(self, frame_index: int):
        if frame_index in self._buffer.get_all_indices():
            qimage, boxes, scores, labels = self._buffer._process_frame(frame_index)
        else:
            qimage, boxes, scores, labels = self._disk._process_frame_disk(frame_index)

        self.frame_ready.emit(qimage, boxes, scores, labels)
        self.frame_index_changed.emit(frame_index)
        self._last_frame_index = frame_index
