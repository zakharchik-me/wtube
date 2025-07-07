import os
import cv2
import glob
from .base import FrameSource
from utils.registry import register
import datetime
from typing import Optional


@register("source")
class VideoFileSource(FrameSource):
    def __init__(self, path: str):
        self.path = path
        self.cap  = cv2.VideoCapture(path)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_ts  =    self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return frame, frame_num, frame_ts

    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> datetime.timedelta:
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 1.0
        seconds = round(self.total_frames / fps)
        return datetime.timedelta(seconds=seconds)


@register("source")
class ImageFolderSource(FrameSource):
    def __init__(self, path):
        self.paths = sorted(glob.glob(os.path.join(path, "*.jpg")))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.paths):
            raise StopIteration
        frame = cv2.imread(self.paths[self.index])
        self.index += 1

        frame_num = int(self.index)
        frame_ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        return frame, frame_num, frame_ts

    def __len__(self):
        return len(self.paths)

    @property
    def total_frames(self) -> Optional[int]:
        return len(self.paths)

@register("source")
class UDPStreamSource(FrameSource):
    def __init__(self, port: int, connection_type: str = 'webcam'):
        if connection_type == 'udp':
            pipeline = (
                f"udpsrc port={port} ! "
                "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! "
                "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
                "video/x-raw ! appsink sync=false"
            )
        elif connection_type == 'rtsp':
            print("Not realized yet")
            pipeline = ()
        elif connection_type == 'webcam':
            pipeline = (
            "v4l2src device=/dev/video0 ! "
            "image/jpeg,width=1280,height=720 ! jpegdec ! tee name=t "
            "t. ! queue ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false drop=true max-buffers=1 name=appsink "
            "t. ! queue ! videoconvert ! "
            "x264enc speed-preset=superfast key-int-max=30 tune=zerolatency ! "
            "h264parse ! rtph264pay config-interval=1 ! "
            f"udpsink host=127.0.0.1 port={port}"
        )
        else:
            raise KeyError(f"Can't be opened such connection type {connection_type}")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть GStreamer pipeline на порту {port}")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise StopIteration

        frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        return frame, frame_num, frame_ts

    def __len__(self):
        return 30 * 600

    @property
    def total_frames(self) -> Optional[int]:
        # None means “unknown”
        return 30 * 600
