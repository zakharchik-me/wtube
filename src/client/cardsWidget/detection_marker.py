from typing import List, Dict, Tuple

from .class_card import ClassCardsManager, set_cards_manager
from utils.paint import load_class_config, build_palette

COCO_CLASSES_PATH = "client/configs/class_config.yml"


class ClassStat:
    def __init__(self, class_id: int, confidence: float = 0.0, frame_num: int = 0):
        self.class_id = class_id
        self.confidence = confidence
        self.frame_num = frame_num
        self.count = 0

    def __repr__(self):
        return (f"ClassStat(class_id={self.class_id}, confidence={self.confidence}, "
                f"frame_num={self.frame_num}, count={self.count})")


class ClassStatsManager:
    def __init__(self, class_names: Dict[int, str], cards_manager: ClassCardsManager):
        self.class_names = class_names
        self.class_stats: Dict[int, ClassStat] = {
            class_id: ClassStat(class_id) for class_id in class_names.keys()
        }
        self.cards_manager = cards_manager

    def update(self, class_id: int, confidence: float, frame_num: int):
        stat = self.class_stats.get(class_id)
        if stat is None:
            stat = ClassStat(class_id)
            self.class_stats[class_id] = stat
        stat.frame_num = frame_num
        stat.count += 1

        self.cards_manager.add_timestamp(class_id, stat.frame_num, confidence)

    def get(self, class_id: int) -> ClassStat:
        return self.class_stats.get(class_id)


coco_classes = load_class_config(COCO_CLASSES_PATH)
palette = build_palette()
cards_manager = ClassCardsManager(coco_classes, palette)
set_cards_manager(cards_manager)
stats_manager = ClassStatsManager(coco_classes, cards_manager)


class DetectionMarker:
    def __init__(self, log: Dict, frame_num: int):
        self.labels, self.confidence = self._load_log(log)
        self.frame_num = frame_num

        for label, conf in zip(self.labels, self.confidence):
            stats_manager.update(label, conf, self.frame_num)

    def _load_log(self, log: Dict) -> Tuple[List[int], List[float]]:
        annotations = log["logs"]["annotations"]
        scores = [ann["score"] for ann in annotations]
        labels = [int(ann["label"]) for ann in annotations]
        return labels, scores
