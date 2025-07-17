from typing import List, Dict, Tuple, Optional

from PyQt6.QtCore import Qt, QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QComboBox, QGroupBox, QScrollArea, QPushButton, QFrame
)
from utils.paint import rgb2hex


class ClassCard:
    def __init__(self, label_id: int, color: str):
        self.label_id = label_id
        self.color = rgb2hex(color)
        self.timestamps: Dict[int, List[int]] = {i: [] for i in range(0, 100)}
        self.include: bool = True
        self.confidence = 30
        self.count = 0

    def add_timestamp(self, timestamp: int, confidence: int):
        self.timestamps[confidence].append(timestamp)
        self.count += 1

    def get_color(self) -> str:
        return self.color

    def get_timestamps_by_conf(self, conf_thr: int) -> List[int]:
        all_ts = []
        for conf, ts_list in self.timestamps.items():
            if conf >= conf_thr:
                all_ts.extend(ts_list)
        self.count = len(all_ts)
        return all_ts

    def set_include(self, include: bool):
        self.include = include

    def is_included(self) -> bool:
        return self.include


class ClassCardsManager(QObject):
    card_updated = pyqtSignal(int)

    def __init__(self, class_names: Dict[int, str], palette: List[Tuple[int, int, int]]):
        super().__init__()
        self.class_names = class_names
        self.cards: Dict[int, ClassCard] = {}
        self.palette = palette
        for class_id in class_names.keys():
            color = self._get_color_for_class_id(class_id)
            self.cards[class_id] = ClassCard(label_id=class_id, color=color)

    def add_timestamp(self, class_id: int, timestamp: int, confidence: float):
        card = self.cards.get(class_id)
        confidence = int(confidence*100)
        if card:
            card.add_timestamp(timestamp, confidence)
            self.card_updated.emit(class_id)

    def all_cards(self) -> List[ClassCard]:
        return list(self.cards.values())

    def _get_color_for_class_id(self, class_id: int) -> Tuple[int, int, int]:
        return self.palette[class_id % len(self.palette)]

    def get_all_highlight_timestamps(self, confidence_threshold: int) -> List[Tuple[int, int]]:
        highlights = []
        for class_id, card in self.cards.items():
            if card.is_included():
                for ts in card.get_timestamps_by_conf(confidence_threshold):
                    highlights.append((class_id, ts))
        return sorted(set(highlights), key=lambda x: (x[1], x[0]))


def set_cards_manager(manager: ClassCardsManager):
    global _shared_cards_manager
    _shared_cards_manager = manager


def get_cards_manager() -> Optional[ClassCardsManager]:
    return _shared_cards_manager


class ClassCardWidget(QWidget):
    include_changed = pyqtSignal(int, bool)
    timestamp_selected = pyqtSignal(int)

    def __init__(self, card: ClassCard, class_name: str):
        super().__init__()
        self.card = card
        self.class_name = class_name
        self.threshold = 30
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        title = QLabel(f"{self.class_name} (ID: {self.card.label_id})")
        title.setStyleSheet(f"font-weight: bold; color: {self.card.get_color()};")
        layout.addWidget(title)

        self.count_label = QLabel(f"Count: {len(self.card.get_timestamps_by_conf(self.threshold))}")
        layout.addWidget(self.count_label)

        self.timestamp_scroll = QScrollArea()
        self.timestamp_scroll.setWidgetResizable(True)
        self.timestamp_scroll.setFixedHeight(80)

        self.timestamp_container = QWidget()
        self.timestamp_layout = QVBoxLayout()
        self.timestamp_layout.setSpacing(2)
        self.timestamp_container.setLayout(self.timestamp_layout)
        self.timestamp_scroll.setWidget(self.timestamp_container)
        layout.addWidget(self.timestamp_scroll)

        self.checkbox = QCheckBox("Include")
        self.checkbox.setChecked(self.card.is_included())
        self.checkbox.stateChanged.connect(self.toggle_include)
        layout.addWidget(self.checkbox)

        self.setLayout(layout)
        self.populate_timestamps()

    def populate_timestamps(self):
        while self.timestamp_layout.count():
            item = self.timestamp_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for ts in sorted(self.card.get_timestamps_by_conf(self.threshold)):
            btn = QPushButton(str(ts))
            btn.setFixedHeight(22)
            btn.setStyleSheet("text-align: left; padding-left: 5px;")
            btn.clicked.connect(lambda _, ts_val=ts: self.timestamp_selected.emit(ts_val))
            self.timestamp_layout.addWidget(btn)

    def toggle_include(self, state: int):
        include = state == Qt.CheckState.Checked.value
        self.card.set_include(include)
        self.include_changed.emit(self.card.label_id, include)

    def update_timestamps(self):
        self.populate_timestamps()
        self.count_label.setText(f"Count: {len(self.card.get_timestamps_by_conf(self.threshold))}")

    def set_confidence_threshold(self, threshold: int):
        self.threshold = threshold
        self.populate_timestamps()
        self.count_label.setText(f"Count: {len(self.card.get_timestamps_by_conf(self.threshold))}")

class ClassCardsWidget(QWidget):
    toggleIncludeCard = pyqtSignal(int, bool)
    timestampClicked = pyqtSignal(int)
    highlight = pyqtSignal()
    pauseRequest = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.card_widgets = {}

        self.manager = get_cards_manager()
        self.manager.card_updated.connect(self.on_card_updated)
        set_cards_manager(self.manager)

        self.layout = QVBoxLayout(self)
        self.class_group = QGroupBox("Classes")
        self.class_layout = QVBoxLayout()
        self.class_layout.setContentsMargins(5, 5, 5, 5)
        self.class_layout.setSpacing(5)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(5)

        self.content_widget.setLayout(self.content_layout)
        self.scroll_area.setWidget(self.content_widget)
        self.class_layout.addWidget(self.scroll_area)
        self.class_group.setLayout(self.class_layout)

        self.layout.addWidget(self.class_group)
        # self.layout.addStretch()

        self.initialize_cards()

    def initialize_cards(self):
        for card in self.manager.all_cards():
            if card.count == 0:
                continue
            class_id = card.label_id
            if class_id not in self.card_widgets:
                class_name = self.manager.class_names[class_id]
                widget = ClassCardWidget(card, class_name)
                widget.include_changed.connect(self.on_include_changed)
                widget.timestamp_selected.connect(self.on_timestamp_clicked)
                self.card_widgets[class_id] = widget
                self.content_layout.addWidget(widget)

    @pyqtSlot(int)
    def on_card_updated(self, class_id: int):
        card = self.manager.cards[class_id]

        if card.count == 0:
            return

        if class_id in self.card_widgets:
            widget = self.card_widgets[class_id]
            widget.update_timestamps()
        else:
            class_name = self.manager.class_names[class_id]
            widget = ClassCardWidget(card, class_name)
            widget.include_changed.connect(self.on_include_changed)
            widget.timestamp_selected.connect(self.on_timestamp_clicked)
            self.card_widgets[class_id] = widget
            self.content_layout.addWidget(widget)

        self.highlight.emit()

    def on_include_changed(self, class_id: int, include: bool):
        self.toggleIncludeCard.emit(class_id, include)

    def update_all_cards_by_confidence(self, threshold: int):
        for widget in self.card_widgets.values():
            widget.set_confidence_threshold(threshold)

    def on_timestamp_clicked(self, timestamp: int):
        self.timestampClicked.emit(timestamp)
        self.pauseRequest.emit()

    def get_highlight_timestamps(self) -> List[Tuple[int, int]]:
        threshold = next(iter(self.card_widgets.values()), None)
        if threshold is not None:
            threshold = threshold.threshold
        else:
            threshold = 30

        return self.manager.get_all_highlight_timestamps(threshold)
