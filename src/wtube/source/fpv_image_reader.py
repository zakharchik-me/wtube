import cv2
import threading
import time
import os
from datetime import datetime, timedelta

# from PyQt5.QtCore import QThread  # or use PySide6


class FPVImageReader:
    def __init__(self, settings, telemetry_type, app_mode,
                 ip: str = '', video_for_log: str = '', video_from_log: bool = False):
        self.settings = settings
        self.telemetry_type = telemetry_type
        self.app_mode = app_mode
        self.ip = ip
        self.video_for_log_dir = video_for_log
        self.video_from_log = video_from_log

        # frames
        self._temp_image = None
        self._original_frame = None
        self._has_new_frame = threading.Event()
        self._stop_event = threading.Event()
        self._mutex = threading.Lock()

        # log playback
        self.images = []
        self.images_times = []
        self.log_index = 0

        # configure basics
        self.resolution = (800, 600)
        self.fps = 60
        self.bitrate = self.settings.udp_stream_bitrate
        self.pipeline = None

        # Open camera / log
        self.init_camera()

        # thread for capture loop
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def init_camera(self):
        # Stop existing
        self._stop_event.set()
        time.sleep(0.01)

        # Prepare sources
        if self.video_from_log:
            # load images
            for fname in sorted(os.listdir(self.video_for_log_dir)):
                if fname.lower().endswith('.jpg'):
                    path = os.path.join(self.video_for_log_dir, fname)
                    timestamp_str = os.path.splitext(fname)[0].split('_')[-1]
                    # parse e.g. hh:mm:ss.zzz
                    try:
                        t = datetime.strptime(timestamp_str, '%H:%M:%S.%f')
                    except ValueError:
                        t = None
                    self.images.append(path)
                    self.images_times.append(t)
            self.settings.log_start = datetime.now()
            self.settings.log_duration = (self.images_times[-1] - self.images_times[0]).total_seconds()
            self.capture = None
        else:
            # realtime or video file
            if self.video_for_log_dir:
                self.capture = cv2.VideoCapture(self.video_for_log_dir)
            else:
                # gstreamer pipeline or device
                pipeline = self._build_pipeline()
                self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        # reset flags
        self._stop_event.clear()
        self._has_new_frame.clear()

    def _build_pipeline(self) -> str:
        # Build GStreamer pipeline string based on telemetry_type and app_mode
        # Example for UDP H264
        if self.telemetry_type == 'ETHERNET':
            return f"udpsrc port={self.settings.udp_stream_port} ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false"
        # Fallback to device
        return self.settings.camera_path

    def _capture_loop(self):
        while not self._stop_event.is_set():
            frame = None
            if self.video_from_log:
                # simulate timing
                now = datetime.now()
                elapsed = (now - self.settings.log_start).total_seconds()
                # find index
                idx = min(int(elapsed * self.fps), len(self.images) - 1)
                if idx != self.log_index:
                    self.log_index = idx
                    frame = cv2.imread(self.images[self.log_index])
            else:
                if self.capture and self.capture.isOpened():
                    ret, img = self.capture.read()
                    if ret:
                        frame = img
            if frame is not None:
                with self._mutex:
                    self._temp_image = frame.copy()
                    self._original_frame = frame.copy()
                    self._has_new_frame.set()
            time.sleep(1.0 / self.fps)

    def has_new_frame(self) -> bool:
        return self._has_new_frame.is_set()

    def get_next_frame(self):
        # blocks until new frame
        self._has_new_frame.wait()
        with self._mutex:
            frame = self._temp_image.copy()
            self._has_new_frame.clear()
        return frame

    def stop(self):
        self._stop_event.set()
        if self.capture:
            self.capture.release()
        self.thread.join(timeout=1)

    def set_camera_path(self, path: str):
        self.settings.camera_path = path
        self.init_camera()

    def set_ip(self, ip: str):
        self.ip = ip
        self.init_camera()

    def reinit(self):
        self.init_camera()

    def switch_camera(self, available_paths: list):
        # cycle through devices
        try:
            idx = available_paths.index(self.settings.camera_path)
            new = available_paths[(idx + 1) % len(available_paths)]
        except ValueError:
            new = available_paths[0]
        self.set_camera_path(new)

    def __del__(self):
        self.stop()
