import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton,
    QComboBox, QFileDialog, QGridLayout, QSizePolicy, QLabel, QSlider, QHBoxLayout as QHBoxLayoutWidgets, QListWidget
)
from PyQt6.QtCore import Qt, pyqtSlot

from app.viewer import ImageViewer
from app.server import DetectionLauncher
from app.client import ClientLauncher
from client.slider.buffered_slider import BufferedSlider
from utils.export import Export
from client.cardsWidget.class_card import ClassCardsWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bounding Box App")
        self.resize(1280, 800)

        self.server_started = False
        self.paused = False
        self.current_frame_buffered = 0

        self.root_path = 'logs/'

        self.detection_launcher = DetectionLauncher()
        self.buffer_launcher = ClientLauncher(viewer=None)
        self.buffer_launcher.frame_changed.connect(self._update_slider_position)
        self.buffer_launcher.log_buffered.connect(self._update_buffer_value)

        self.chosen_video_file = None

        self.pause_on_slider = False

        self.jump_triggered_pause = False

        self._setup_ui()

    def _setup_ui(self) -> None:
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._init_video_group(), 5)
        layout.addWidget(self._init_control_group(), 1)

        container.setLayout(layout)
        self.setCentralWidget(container)

        self.buffer_launcher.viewer = self.viewer

    def _init_video_group(self) -> QGroupBox:
        group = QGroupBox("Video Stream")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.viewer = ImageViewer()
        self.viewer.setFixedSize(1280, 720)
        layout.addWidget(self.viewer)

        slider_controls_layout = QHBoxLayout()
        slider_controls_layout.setContentsMargins(5, 5, 5, 5)
        slider_controls_layout.setSpacing(5)

        slider_controls_layout.addStretch()

        self.seek_to_buffer_btn = QPushButton("⇥")
        self.seek_to_buffer_btn.setFixedSize(30, 24)
        self.seek_to_buffer_btn.setToolTip("Jump to buffered position")
        self.seek_to_buffer_btn.clicked.connect(self._seek_to_buffered_position)
        self.seek_to_buffer_btn.setEnabled(False)

        slider_controls_layout.addWidget(self.seek_to_buffer_btn)
        layout.addLayout(slider_controls_layout)

        self.seek_slider = BufferedSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setMinimum(1)
        self.seek_slider.setSingleStep(1)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.pauseRequested.connect(self._pause_on_slider_press)
        self.seek_slider.highlightClicked.connect(self._on_jump_to_frame)
        layout.addWidget(self.seek_slider)

        group.setLayout(layout)
        return group

    def _init_control_group(self) -> QGroupBox:
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.conf_label = QLabel()
        self.conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_manager = ClassCardsWidget()
        self.conf_slider = self._create_slider()
        self.cards_manager.highlight.connect(self._update_slider_highlights)
        layout.addWidget(self.conf_label)
        layout.addWidget(self.conf_slider)

        conf_jump_layout = QHBoxLayout()
        for val in [0, 30, 50, 90]:
            btn = QPushButton(str(val))
            btn.setFixedSize(30, 15)
            btn.clicked.connect(lambda checked, v=val: self.conf_slider.setValue(v))
            conf_jump_layout.addWidget(btn)
        layout.addLayout(conf_jump_layout)

        self.cards_manager.toggleIncludeCard.connect(self._toggle_include)
        layout.addWidget(self.cards_manager, stretch=1)
        self.cards_manager.timestampClicked.connect(self._on_jump_to_frame)
        self.cards_manager.pauseRequest.connect(self._pause_on_ts)

        self.config_combo = self._create_config_combo()
        layout.addWidget(self.config_combo)

        file_layout = QHBoxLayoutWidgets()
        self.choose_file_btn = QPushButton("Choose File")
        self.choose_file_btn.setEnabled(False)
        self.choose_file_btn.clicked.connect(self._choose_video_file)
        file_layout.addWidget(QLabel("Video File:"))
        file_layout.addWidget(self.choose_file_btn)
        layout.addLayout(file_layout)

        layout.addLayout(self._create_buttons())
        layout.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export")
        self.export_all_btn = QPushButton("Export All")

        self.export_btn.setEnabled(False)
        self.export_all_btn.setEnabled(False)

        self.export_btn.clicked.connect(self.export_single)
        self.export_all_btn.clicked.connect(self.export_all)

        layout.addWidget(self.export_btn)
        layout.addWidget(self.export_all_btn)

        group.setLayout(layout)
        group.setMaximumWidth(500)

        self.config_combo.currentIndexChanged.connect(self._on_config_changed)

        return group

    def _create_slider(self) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setSingleStep(5)
        slider.setPageStep(5)
        slider.setTickInterval(5)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setValue(30)
        slider.valueChanged.connect(self._on_confidence_changed)
        self._on_confidence_changed(slider.value())
        return slider

    def _create_config_combo(self) -> QComboBox:
        combo = QComboBox()
        for cfg in sorted(Path("client/configs/source").glob("*.yml")):
            combo.addItem(cfg.name, str(cfg))
        return combo

    def _create_buttons(self) -> QGridLayout:
        layout = QGridLayout()
        layout.setSpacing(5)

        self.start_btn = QPushButton("Start Server")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.restart_btn = QPushButton("Restart Client")

        self.pause_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)

        for btn in (self.start_btn, self.pause_btn, self.stop_btn, self.restart_btn):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: red;
                color: white;
            }
            QPushButton:hover {
                background-color: darkred;
            }
        """)

        layout.addWidget(self.start_btn, 0, 0)
        layout.addWidget(self.pause_btn, 0, 1)
        layout.addWidget(self.restart_btn, 1, 0, 1, 2)

        self.start_btn.clicked.connect(self._start_server)
        self.pause_btn.clicked.connect(self._toggle_pause_resume)
        self.stop_btn.clicked.connect(self._stop_and_exit)
        self.restart_btn.clicked.connect(self._restart_client)

        return layout

    def _start_server(self) -> None:
        self.detection_launcher.start()
        self.server_started = True

        self.start_btn.setText("Start Client")
        self.start_btn.clicked.disconnect()
        self.start_btn.clicked.connect(self._start_client)

    def _start_client(self) -> None:
        if not self.server_started:
            return

        from utils.config import load_config
        config_path = self.config_combo.currentData()
        cfg = load_config(config_path)

        if cfg["source"]["name"] == "VideoFileSource":
            if not self.chosen_video_file:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select video file", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
                )
                if not file_path:
                    return
                self.chosen_video_file = file_path
                self.choose_file_btn.setText(f"File: {Path(file_path).name}")

            cfg["source"]["args"]["path"] = self.chosen_video_file

        if self.buffer_launcher is None:
            self.buffer_launcher = ClientLauncher(viewer=self.viewer)
            self.buffer_launcher.frame_changed.connect(self._update_slider_position)
            self.buffer_launcher.frame_buffered.connect(self._update_buffer_value)
        else:
            self.buffer_launcher.stop()

        self.buffer_launcher.start(config_path=None, config_dict=cfg)

        self.seek_slider.setEnabled(True)
        self.seek_slider.setValue(1)
        self.seek_slider.setMaximum(self.buffer_launcher.source.total_frames)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self.config_combo.setEnabled(False)

        self.export_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)

    def _restart_client(self):
        if self.detection_launcher:
            self.detection_launcher.stop()

        if self.buffer_launcher:
            self.buffer_launcher.stop()

        self.chosen_video_file = None

        self._start_server()

        self._start_client()

        if self.buffer_launcher and hasattr(self.buffer_launcher, 'source') and hasattr(self.buffer_launcher.source,
                                                                                        'total_frames'):
            total_frames = self.buffer_launcher.source.total_frames
        else:
            total_frames = 0
            print("Warning: buffer_launcher.source.total_frames not available")

        self.seek_slider.setEnabled(True)
        self.seek_slider.setValue(1)
        self.seek_slider.setMaximum(total_frames)

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.config_combo.setEnabled(False)

        self._update_export_buttons_enabled()

    def _toggle_pause_resume(self) -> None:
        if self.paused:
            self.buffer_launcher.play()
            self.pause_btn.setText("Pause")
        else:
            self.buffer_launcher.pause()
            self.pause_btn.setText("Resume")
        self.paused = not self.paused
        self._update_export_buttons_enabled()

    def _stop_and_exit(self) -> None:
        self.detection_launcher.stop()
        self.buffer_launcher.stop()
        self.close()

    def _on_slider_released(self) -> None:
        new_index = self.seek_slider.value()

        if new_index > self.current_frame_buffered:
            new_index = self.current_frame_buffered
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(new_index)
            self.seek_slider.blockSignals(False)

        self.buffer_launcher.frame_worker.seek(new_index)

    def _toggle_include(self, class_id: int, include: bool) -> None:
        self.viewer.set_include_class(class_id, include)

    def _on_confidence_changed(self, value: int) -> None:
        threshold = (value - value%5) / 100
        self.conf_label.setText(f"Confidence: {int(threshold * 100)}%")
        self.viewer.set_confidence_threshold(threshold)
        self.cards_manager.update_all_cards_by_confidence(value)

    def _update_slider_position(self, frame_index: int) -> None:
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_index)
        self.seek_slider.blockSignals(False)
        self._update_seek_button_enabled()
        self._update_export_buttons_enabled()

    def _update_buffer_value(self, buffer_value: int) -> None:
        self.seek_slider.setBufferValue(buffer_value)
        self.current_frame_buffered = buffer_value
        self._update_seek_button_enabled()

    @pyqtSlot()
    def _choose_video_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select video file", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.chosen_video_file = file_path
            self.choose_file_btn.setText(f"File: {Path(file_path).name}")

    @pyqtSlot()
    def _on_config_changed(self) -> None:
        from utils.config import load_config
        config_path = self.config_combo.currentData()
        if not config_path:
            self.choose_file_btn.setEnabled(False)
            self.chosen_video_file = None
            self.choose_file_btn.setText("Choose File")
            return
        cfg = load_config(config_path)
        if cfg["source"]["name"] == "VideoFileSource":
            self.choose_file_btn.setEnabled(True)
        else:
            self.choose_file_btn.setEnabled(False)
            self.chosen_video_file = None
            self.choose_file_btn.setText("Choose File")

    def _pause_on_slider_press(self) -> None:
        if not self.paused:
            self._toggle_pause_resume()

    def _pause_on_ts(self) -> None:
        if not self.paused:
            self._toggle_pause_resume()

    def _update_seek_button_enabled(self) -> None:
        if self.seek_slider.value() < self.current_frame_buffered:
            self.seek_to_buffer_btn.setEnabled(True)
        else:
            self.seek_to_buffer_btn.setEnabled(False)

    def _seek_to_buffered_position(self) -> None:
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self.current_frame_buffered)
        self.seek_slider.blockSignals(False)

        self.buffer_launcher.frame_worker.seek(self.current_frame_buffered)

        if not self.paused:
            self.buffer_launcher.play()

        self._update_export_buttons_enabled()

    def _update_export_buttons_enabled(self) -> None:
        enabled = self.paused or self.seek_slider.value() >= self.current_frame_buffered
        self.export_btn.setEnabled(enabled)
        self.export_all_btn.setEnabled(enabled)

    @pyqtSlot()
    def export_single(self) -> None:
        if not self.buffer_launcher or not hasattr(self.buffer_launcher, "source"):
            print("No buffer_launcher or source available for export.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Exported Video (from current stream)", "", "MP4 Video (*.mp4)"
        )
        if not save_path:
            return

        path = os.path.join(self.root_path, os.path.splitext(os.path.basename(self.chosen_video_file))[0])

        export = Export(
            source=self.buffer_launcher.source,
            path=path,
            output_path=save_path,
        )
        export.run()

    @pyqtSlot()
    def export_all(self) -> None:
        if not os.path.exists(self.root_path):
            print(f"Root path '{self.root_path}' does not exist.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save All Streams"
        )
        if not output_dir:
            return

        subdirs = [d for d in Path(self.root_path).iterdir() if d.is_dir()]

        if not subdirs:
            print(f"No subdirectories found in '{self.root_path}'.")
            return

        for subdir in subdirs:
            video_name = subdir.name
            save_path = Path(output_dir) / f"{video_name}.mp4"

            source_path = str(subdir)

            try:
                export = Export(
                    source=None,
                    path=source_path,
                    output_path=str(save_path)
                )
                export.run()
                print(f"Exported: {save_path}")
            except Exception as e:
                print(f"Failed to export {video_name}: {e}")

    def _on_jump_to_frame(self, frame: int) -> None:
        if frame <= self.current_frame_buffered:
            self.seek_slider.setValue(frame)
            self.buffer_launcher.frame_worker.seek(frame)
            self.paused = False
            self._toggle_pause_resume()

    def _update_slider_highlights(self):
        highlights = self.cards_manager.get_highlight_timestamps()
        self.seek_slider.set_highlights(highlights)

