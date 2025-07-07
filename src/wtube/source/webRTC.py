import asyncio
import threading
import queue

import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.signaling import BYE
from .base import FrameSource

class WebRTCSource(FrameSource):
    """
    FrameSource implementation that receives video frames over a WebRTC connection

    Parameters:
        offer_sdp: str          -- the SDP offer from the remote peer
        offer_type: str         -- usually "offer" (default)
        signaling: Signaling    -- an aiortc signaling instance for ICE negotiation
    """
    def __init__(self, offer_sdp: str, offer_type: str = "offer", signaling=None):
        self.pc = RTCPeerConnection()
        self.queue = queue.Queue()
        self.loop = asyncio.new_event_loop()

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.run_coroutine_threadsafe(self._recv_video(track), self.loop)

        def _start_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._negotiate(offer_sdp, offer_type, signaling))
            self.loop.run_forever()

        t = threading.Thread(target=_start_loop, daemon=True)
        t.start()

    async def _negotiate(self, sdp: str, sdp_type: str, signaling):
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp, sdp_type))
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        if signaling:
            await signaling.send(self.pc.localDescription)
            while True:
                obj = await signaling.receive()
                if obj is None or obj == BYE:
                    break
                await self.pc.addIceCandidate(obj)

    async def _recv_video(self, track):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            self.queue.put(img)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            frame = self.queue.get(timeout=5)
        except queue.Empty:
            raise StopIteration
        return frame

    def close(self):
        def stopper():
            self.loop.stop()
        asyncio.run_coroutine_threadsafe(self.pc.close(), self.loop)
        self.loop.call_soon_threadsafe(stopper)
