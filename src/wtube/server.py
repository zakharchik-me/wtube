import sys
import threading
from queue import Queue
from flask import Flask, request
from PyQt6.QtCore import QThread
from .server_worker import DetectionWorker

frame_queue = Queue()

# Flask-сервер
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    frame_num = request.form.get("frame_num")
    timestamp = request.form.get("timestamp")

    if file:
        file_bytes = file.read()
        frame_queue.put((file_bytes, int(frame_num), float(timestamp)))
        return "OK", 200

    return "Missing image", 400


def start_flask_server():
    app.run(host="0.0.0.0", port=8000, threaded=True)