from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np

from app.config import AppConfig
from app.database import EventStore
from app.domain import MonitorRules, ThreatAssessment, ThreatEvaluator, Zone
from app.sms import SMSNotifier
from app.vision import MotionDetector, ObjectDetector, VisionResult, draw_overlay

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FramePacket:
    sequence: int
    captured_at: datetime
    frame: np.ndarray


@dataclass(slots=True)
class DetectionPacket:
    source: FramePacket
    result: VisionResult


@dataclass(slots=True)
class AlertJob:
    event_id: int
    body: str


def _replace_latest(target: queue.Queue, value: object) -> None:
    try:
        target.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    target.put_nowait(value)


class SecurityMonitor:
    def __init__(self, config: AppConfig, store: EventStore):
        self.config = config
        self.store = store
        self.notifier = SMSNotifier(config)
        self.evaluator = ThreatEvaluator()
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector(config.yolo_model, config.yolo_confidence)

        self.capture_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self.analysis_queue: queue.Queue[DetectionPacket] = queue.Queue(maxsize=2)
        self.alert_queue: queue.Queue[AlertJob] = queue.Queue(maxsize=20)
        self._stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._threads: list[threading.Thread] = []
        self._last_event_at = 0.0
        self._last_alert_at = 0.0
        self._status_lock = threading.Lock()
        self._status = "stopped"
        self._error: str | None = None
        self._fps = 0.0
        self._last_assessment: ThreatAssessment | None = None

        default_zones = [{"name": "Main area", "points": [[0.1, 0.15], [0.9, 0.15], [0.9, 0.9], [0.1, 0.9]], "enabled": True}]
        self.zones = self._decode_zones(store.get_setting("zones", default_zones))
        saved_rules = store.get_setting("rules", {})
        self.rules = MonitorRules(
            schedule_enabled=bool(saved_rules.get("schedule_enabled", False)),
            active_from=str(saved_rules.get("active_from", "22:00")),
            active_until=str(saved_rules.get("active_until", "06:00")),
            watch_labels=tuple(saved_rules.get("watch_labels", ["person", "car", "truck", "dog"])),
        )
        self.armed = bool(store.get_setting("armed", True))

    @staticmethod
    def _decode_zones(items: list[dict]) -> list[Zone]:
        return [
            Zone(str(item["name"]), tuple(tuple(float(v) for v in point) for point in item["points"]), bool(item.get("enabled", True)))
            for item in items
        ]

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True),
            threading.Thread(target=self._detection_loop, name="vision-detection", daemon=True),
            threading.Thread(target=self._analysis_loop, name="threat-analysis", daemon=True),
            threading.Thread(target=self._alert_loop, name="sms-alerts", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3)
        with self._status_lock:
            self._status = "stopped"

    def set_armed(self, armed: bool) -> None:
        self.armed = armed
        self.store.set_setting("armed", armed)

    def set_zones(self, zones: list[Zone]) -> None:
        self.zones = zones
        self.store.set_setting("zones", [zone.as_dict() for zone in zones])

    def set_rules(self, rules: MonitorRules) -> None:
        self.rules = rules
        self.store.set_setting(
            "rules",
            {
                "schedule_enabled": rules.schedule_enabled,
                "active_from": rules.active_from,
                "active_until": rules.active_until,
                "watch_labels": list(rules.watch_labels),
            },
        )

    def status(self) -> dict:
        with self._status_lock:
            assessment = self._last_assessment
            return {
                "state": self._status,
                "armed": self.armed,
                "error": self._error,
                "fps": round(self._fps, 1),
                "model": self.config.yolo_model,
                "model_error": self.object_detector.error,
                "sms_mode": "dry-run" if self.config.sms_dry_run else ("ready" if self.notifier.configured else "not-configured"),
                "assessment": None if assessment is None else {
                    "score": assessment.score,
                    "level": assessment.level,
                    "summary": assessment.summary,
                },
            }

    def jpeg_stream(self):
        last_frame: bytes | None = None
        while not self._stop.is_set():
            with self._frame_ready:
                self._frame_ready.wait_for(
                    lambda previous=last_frame: self._latest_jpeg is not previous
                    or self._stop.is_set(),
                    timeout=2,
                )
                frame = self._latest_jpeg
            if frame is None or frame is last_frame:
                continue
            last_frame = frame
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def queue_test_alert(self) -> str:
        result = self.notifier.send("Watchtower Vision test: SMS alerts are connected and ready.")
        return result.status

    def _capture_loop(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            with self._status_lock:
                self._status = "camera-error"
                self._error = f"Could not open webcam index {self.config.camera_index}"
            capture.release()
            return

        with self._status_lock:
            self._status = "running"
            self._error = None
        sequence = 0
        started = time.monotonic()
        while not self._stop.is_set():
            ok, frame = capture.read()
            if not ok:
                with self._status_lock:
                    self._status = "camera-error"
                    self._error = "The webcam stopped returning frames"
                break
            sequence += 1
            _replace_latest(self.capture_queue, FramePacket(sequence, datetime.now(UTC), frame))
            elapsed = time.monotonic() - started
            if elapsed >= 1:
                with self._status_lock:
                    self._fps = sequence / elapsed
        capture.release()

    def _detection_loop(self) -> None:
        cached_detections = []
        while not self._stop.is_set():
            try:
                packet = self.capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            motion_ratio, motion_boxes = self.motion_detector.detect(packet.frame)
            if packet.sequence % self.config.detection_stride == 0:
                cached_detections = self.object_detector.detect(packet.frame)
            result = VisionResult(list(cached_detections), motion_ratio, motion_boxes)
            _replace_latest(self.analysis_queue, DetectionPacket(packet, result))

    def _analysis_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.analysis_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            frame = packet.source.frame
            height, width = frame.shape[:2]
            assessment = self.evaluator.evaluate(
                packet.result.detections,
                packet.result.motion_ratio,
                self.zones,
                (width, height),
                self.rules,
                packet.source.captured_at.astimezone(),
            )
            self._last_assessment = assessment
            annotated = draw_overlay(frame, packet.result, assessment, self.zones)
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                with self._frame_ready:
                    self._latest_jpeg = encoded.tobytes()
                    self._frame_ready.notify_all()

            now = time.monotonic()
            if self.armed and assessment.score >= self.config.event_threshold and now - self._last_event_at >= 5:
                self._last_event_at = now
                self._record_event(packet, assessment, annotated, now)

    def _record_event(
        self,
        packet: DetectionPacket,
        assessment: ThreatAssessment,
        annotated: np.ndarray,
        now: float,
    ) -> None:
        timestamp = packet.source.captured_at
        filename = f"{timestamp.strftime('%Y%m%dT%H%M%S_%f')}.jpg"
        path = self.config.snapshot_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), annotated)

        should_alert = assessment.score >= self.config.alert_threshold and now - self._last_alert_at >= self.config.alert_cooldown_seconds
        sms_status = "pending" if should_alert else "not-requested"
        event_id = self.store.add_event(
            created_at=timestamp.isoformat(),
            level=assessment.level,
            score=assessment.score,
            summary=assessment.summary,
            snapshot=f"/snapshots/{filename}",
            detections=[item.as_dict() for item in packet.result.detections],
            sms_status=sms_status,
        )
        if should_alert:
            self._last_alert_at = now
            body = f"Watchtower alert ({assessment.score}/100): {assessment.summary}. Time: {timestamp.astimezone().strftime('%Y-%m-%d %H:%M:%S')}"
            try:
                self.alert_queue.put_nowait(AlertJob(event_id, body))
            except queue.Full:
                self.store.update_sms_status(event_id, "queue-full")

    def _alert_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.notifier.send(job.body)
                self.store.update_sms_status(job.event_id, result.status)
            except Exception as exc:
                logger.exception("SMS delivery failed")
                self.store.update_sms_status(job.event_id, f"failed: {type(exc).__name__}")
