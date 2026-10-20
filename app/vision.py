from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.domain import Detection, ThreatAssessment, Zone

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisionResult:
    detections: list[Detection]
    motion_ratio: float
    motion_boxes: list[tuple[int, int, int, int]]


class MotionDetector:
    def __init__(self, minimum_area: int = 900):
        self.minimum_area = minimum_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=350, varThreshold=32, detectShadows=True
        )

    def detect(self, frame: np.ndarray) -> tuple[float, list[tuple[int, int, int, int]]]:
        mask = self._subtractor.apply(frame)
        mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[tuple[int, int, int, int]] = []
        moving_area = 0
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append((x, y, x + width, y + height))
            moving_area += area
        frame_area = max(1, frame.shape[0] * frame.shape[1])
        return min(1.0, moving_area / frame_area), boxes


class ObjectDetector:
    def __init__(self, model_name: str, confidence: float):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self.error: str | None = None

    def _load(self) -> None:
        if self._model is not None or self.error:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        except Exception as exc:  # Model download and hardware errors should not kill motion detection.
            self.error = str(exc)
            logger.exception("YOLO could not be loaded")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._load()
        if self._model is None:
            return []
        results = self._model.predict(frame, conf=self.confidence, verbose=False, imgsz=640)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                detections.append(
                    Detection(
                        label=str(names[class_id]),
                        confidence=float(box.conf[0].item()),
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections


def draw_overlay(
    frame: np.ndarray,
    result: VisionResult,
    assessment: ThreatAssessment,
    zones: list[Zone],
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    palette = {"low": (80, 180, 90), "medium": (40, 190, 230), "high": (30, 120, 240), "critical": (45, 45, 230)}
    color = palette[assessment.level]

    for zone in zones:
        points = np.array(
            [[int(x * width), int(y * height)] for x, y in zone.points], dtype=np.int32
        )
        cv2.polylines(canvas, [points], True, (230, 180, 50), 2)
        cv2.putText(canvas, zone.name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 180, 50), 2)

    for detection in result.detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.0%}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    for x1, y1, x2, y2 in result.motion_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (140, 140, 140), 1)

    banner = f"{assessment.level.upper()}  {assessment.score}/100  {assessment.summary}"
    cv2.rectangle(canvas, (0, 0), (width, 34), (18, 24, 35), -1)
    cv2.putText(canvas, banner, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 247, 250), 2)
    return canvas

