import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pytest

from app.config import AppConfig
from app.database import EventStore
from app.domain import Detection, MonitorRules, ThreatAssessment, ThreatEvaluator, Zone
from app.monitor import AlertJob, DetectionPacket, FramePacket, SecurityMonitor, _replace_latest
from app.sms import SMSNotifier
from app.vision import MotionDetector, ObjectDetector, VisionResult, draw_overlay


@pytest.fixture
def monitor(tmp_path):
    config = AppConfig(data_dir=tmp_path, auto_start=False)
    store = EventStore(config.database_path)
    store.initialize()
    return SecurityMonitor(config, store)


@pytest.mark.parametrize(
    "hour,expected",
    [(0, False), (6, False), (7, True), (12, True), (21, True), (22, False), (23, False)],
)
def test_schedule_overnight(hour, expected):
    assert MonitorRules(True).is_after_hours(datetime(2026, 1, 1, hour)) is expected


@pytest.mark.parametrize(
    "hour,expected",
    [(0, True), (8, True), (9, False), (12, False), (17, False), (18, True), (23, True)],
)
def test_schedule_daytime(hour, expected):
    assert (
        MonitorRules(True, "09:00", "17:00").is_after_hours(datetime(2026, 1, 1, hour)) is expected
    )


@pytest.mark.parametrize("count,motion", [(0, 0), (0, -1), (0, 1), (1, 0), (4, 0.1), (100, 10)])
def test_score_bounds(count, motion):
    a = ThreatEvaluator().evaluate(
        [Detection("person", 0.9, (1, 1, 10, 10))] * count,
        motion,
        [],
        (100, 100),
        MonitorRules(),
        datetime.now(),
    )
    assert 0 <= a.score <= 100
    if count == 0:
        assert a.score <= 30


def test_watch_filter_and_disabled_zone():
    a = ThreatEvaluator().evaluate(
        [Detection("cat", 0.99, (1, 1, 10, 10))], 0, [], (100, 100), MonitorRules(), datetime.now()
    )
    assert a.score == 0
    a = ThreatEvaluator().evaluate(
        [Detection("person", 0.99, (1, 1, 10, 10))],
        0,
        [Zone("x", ((0, 0), (1, 0), (1, 1), (0, 1)), False)],
        (100, 100),
        MonitorRules(),
        datetime.now(),
    )
    assert a.score == 38 and not a.inside_zones


def test_database_concurrency(monitor):
    s = monitor.store

    def insert(i):
        return s.add_event(
            created_at=datetime.now(UTC).isoformat(),
            level="high",
            score=70,
            summary=f"event {i}'; DROP TABLE events;--",
            snapshot=None,
            detections=[],
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(insert, range(100)))
    assert len(set(ids)) == 100
    assert s.stats() == {"total": 100, "high_risk": 100, "today": 100}
    assert len(s.recent_events(10000)) == 100
    assert len(s.recent_events(0)) == 1
    assert s.recent_events()[0]["id"] == max(ids)


def test_persistence(monitor):
    monitor.set_armed(False)
    monitor.set_zones([])
    monitor.set_rules(MonitorRules(True, "09:00", "17:00", ("person",)))
    restored = SecurityMonitor(monitor.config, monitor.store)
    assert restored.armed is False and restored.zones == []
    assert restored.rules == monitor.rules


def test_replace_latest():
    q = queue.Queue(2)
    for i in range(100):
        _replace_latest(q, i)
    assert [q.get_nowait(), q.get_nowait()] == [98, 99]


def test_motion_static_and_change():
    d = MotionDetector()
    frame = np.zeros((240, 320, 3), np.uint8)
    for _ in range(20):
        ratio, boxes = d.detect(frame)
    assert ratio == 0 and boxes == []
    changed = frame.copy()
    changed[50:150, 50:150] = 255
    ratio, boxes = d.detect(changed)
    assert 0 < ratio <= 1 and boxes


def test_overlay_does_not_modify_input():
    frame = np.zeros((240, 320, 3), np.uint8)
    output = draw_overlay(
        frame,
        VisionResult([Detection("person", 0.9, (30, 40, 100, 120))], 0.1, [(10, 10, 50, 50)]),
        ThreatAssessment(80, "critical", "test", (), False),
        [],
    )
    assert output.shape == frame.shape and np.any(output) and not np.any(frame)


def test_model_load_failure_is_safe(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ultralytics", None)
    d = ObjectDetector("missing.pt", 0.5)
    assert d.detect(np.zeros((32, 32, 3), np.uint8)) == [] and d.error


def test_model_conversion():
    d = ObjectDetector("fake", 0.5)
    box = Mock()
    box.xyxy = [np.array([1.2, 2.9, 10.1, 20.8])]
    box.cls = [np.array(0)]
    box.conf = [np.array(0.8)]
    result = Mock()
    result.names = {0: "person"}
    result.boxes = [box]
    d._model = Mock()
    d._model.predict.return_value = [result]
    assert d.detect(np.zeros((32, 32, 3), np.uint8)) == [Detection("person", 0.8, (1, 2, 10, 20))]


def packet():
    f = np.zeros((240, 320, 3), np.uint8)
    return DetectionPacket(
        FramePacket(3, datetime.now(UTC), f),
        VisionResult([Detection("person", 0.9, (50, 50, 150, 200))], 0.1, []),
    )


def test_event_snapshot_cooldown(monitor):
    p = packet()
    a = ThreatAssessment(85, "critical", "test", (), False)
    monitor._record_event(p, a, p.source.frame, 100)
    assert monitor.store.recent_events()[0]["sms_status"] == "pending"
    assert monitor.alert_queue.qsize() == 1
    assert len(list(monitor.config.snapshot_dir.glob("*.jpg"))) == 1
    monitor._record_event(p, a, p.source.frame, 101)
    assert monitor.store.recent_events()[0]["sms_status"] == "not-requested"
    assert monitor.alert_queue.qsize() == 1


def test_alert_queue_full(monitor):
    for i in range(20):
        monitor.alert_queue.put_nowait(AlertJob(i, "x"))
    p = packet()
    monitor._record_event(
        p, ThreatAssessment(90, "critical", "test", (), False), p.source.frame, 100
    )
    assert monitor.store.recent_events()[0]["sms_status"] == "queue-full"


def test_alert_worker_failure_and_recovery(monitor):
    p = packet()
    monitor._record_event(
        p, ThreatAssessment(90, "critical", "test", (), False), p.source.frame, 100
    )
    monitor.notifier.send = Mock(side_effect=RuntimeError("test"))
    t = threading.Thread(target=monitor._alert_loop)
    t.start()
    deadline = time.monotonic() + 2
    while (
        monitor.store.recent_events()[0]["sms_status"] == "pending" and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    monitor._stop.set()
    t.join(2)
    assert monitor.store.recent_events()[0]["sms_status"] == "failed: RuntimeError"
    assert not t.is_alive()


def test_camera_unavailable(monitor, monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr("app.monitor.cv2.VideoCapture", lambda _: capture)
    monitor._capture_loop()
    assert monitor.status()["state"] == "camera-error"
    capture.release.assert_called_once()


def test_stream(monitor):
    monitor._latest_jpeg = b"jpeg"
    stream = monitor.jpeg_stream()
    assert next(stream) == b"--frame\r\nContent-Type: image/jpeg\r\n\r\njpeg\r\n"
    monitor.stop()
    stream.close()


def test_sms_provider_mock(monkeypatch):
    client = Mock()
    client.messages.create.return_value.sid = "fake-id"
    monkeypatch.setattr("twilio.rest.Client", lambda *args: client)
    notifier = SMSNotifier(
        AppConfig(
            sms_dry_run=False,
            twilio_account_sid="fake",
            twilio_auth_token="fake",
            twilio_from_number="+100",
            alert_to_number="+200",
        )
    )
    assert notifier.send("body").message_id == "fake-id"
    client.messages.create.assert_called_once_with(body="body", from_="+100", to="+200")


@pytest.fixture
def api(monitor, monkeypatch):
    monkeypatch.setenv("AUTO_START", "false")
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main, "monitor", monitor)
    monkeypatch.setattr(main, "store", monitor.store)
    with TestClient(main.app) as client:
        yield client


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/healthz",
        "/docs",
        "/openapi.json",
        "/static/dashboard.js",
        "/static/styles.css",
        "/api/status",
        "/api/events",
        "/api/stats",
        "/api/zones",
        "/api/rules",
    ],
)
def test_endpoints(api, path):
    assert api.get(path).status_code == 200


@pytest.mark.parametrize("value", [0, -1, 201, "bad"])
def test_event_limit_validation(api, value):
    assert api.get(f"/api/events?limit={value}").status_code == 422


@pytest.mark.parametrize(
    "zones",
    [
        [{"name": "", "points": [[0, 0], [1, 0], [0, 1]]}],
        [{"name": "x", "points": [[0, 0], [1, 0]]}],
        [{"name": "x", "points": [[0, 0], [2, 0], [0, 1]]}],
        [{"name": "x", "points": [[0, 0], [1, 0], [0, 1]]}] * 9,
    ],
)
def test_invalid_zones(api, zones):
    assert api.put("/api/zones", json={"zones": zones}).status_code == 422


@pytest.mark.parametrize("value", ["25:00", "12:60", "bad", "1:2:3", ""])
def test_invalid_rules(api, value):
    assert api.put("/api/rules", json={"active_from": value}).status_code == 422


def test_api_mutations(api):
    assert api.post("/api/monitor/arm", json={"armed": False}).json() == {"armed": False}
    assert api.get("/api/status").json()["armed"] is False
    assert api.put("/api/zones", json={"zones": []}).json() == {"zones": []}
    assert api.put("/api/rules", json={"watch_labels": [" PERSON "]}).status_code == 200
    assert api.get("/api/rules").json()["watch_labels"] == ["person"]
    assert api.put("/api/rules", json={"watch_labels": [" "]}).status_code == 422
    assert api.post("/api/alerts/test").json() == {"status": "dry-run"}


def test_sms_failure_api(api, monitor):
    monitor.notifier.send = Mock(side_effect=RuntimeError("private token"))
    response = api.post("/api/alerts/test")
    assert response.status_code == 502 and "private token" not in response.text


# Expected robustness requirements: failures expose defects, not test accommodations.
def test_inference_failure_degrades_gracefully():
    d = ObjectDetector("fake", 0.5)
    d._model = Mock()
    d._model.predict.side_effect = RuntimeError("inference failed")
    assert d.detect(np.zeros((32, 32, 3), np.uint8)) == []


def test_failed_snapshot_not_advertised(monitor, monkeypatch):
    monkeypatch.setattr("app.monitor.cv2.imwrite", lambda *args: False)
    p = packet()
    monitor._record_event(
        p, ThreatAssessment(80, "critical", "test", (), False), p.source.frame, 100
    )
    assert monitor.store.recent_events()[0]["snapshot"] is None


def test_rules_response_matches_persisted_state(api):
    response = api.put("/api/rules", json={"watch_labels": [" PERSON ", "person"]})
    assert response.json() == api.get("/api/rules").json()


def test_degenerate_zone_rejected(api):
    assert (
        api.put(
            "/api/zones",
            json={"zones": [{"name": "invalid", "points": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]}]},
        ).status_code
        == 422
    )


def test_threaded_pipeline_end_to_end(monitor, monkeypatch):
    class Camera:
        released = False

        def set(self, *args):
            pass

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.005)
            return True, np.zeros((240, 320, 3), np.uint8)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr("app.monitor.cv2.VideoCapture", lambda _: camera)
    monitor.object_detector.detect = Mock(
        return_value=[Detection("person", 0.9, (50, 50, 150, 200))]
    )
    monitor.motion_detector.detect = Mock(return_value=(0.1, []))
    monitor.start()
    original_threads = list(monitor._threads)
    monitor.start()
    assert monitor._threads == original_threads
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        if (
            monitor.store.stats()["total"]
            and monitor.store.recent_events()[0]["sms_status"] == "dry-run"
        ):
            break
        time.sleep(0.02)
    monitor.stop()
    assert monitor.store.stats()["total"] == 1
    assert monitor.store.recent_events()[0]["sms_status"] == "dry-run"
    assert monitor._latest_jpeg.startswith(b"\xff\xd8")
    assert camera.released and all(not t.is_alive() for t in original_threads)
    assert monitor.status()["state"] == "stopped"


def test_disarmed_pipeline_no_events(monitor, monkeypatch):
    monitor.set_armed(False)
    p = packet()
    monitor.analysis_queue.put(p)
    t = threading.Thread(target=monitor._analysis_loop)
    t.start()
    deadline = time.monotonic() + 2
    while monitor._latest_jpeg is None and time.monotonic() < deadline:
        time.sleep(0.01)
    monitor._stop.set()
    t.join(2)
    assert monitor._latest_jpeg and monitor.store.stats()["total"] == 0


def test_alert_success_worker(monitor):
    p = packet()
    monitor._record_event(
        p, ThreatAssessment(90, "critical", "test", (), False), p.source.frame, 100
    )
    t = threading.Thread(target=monitor._alert_loop)
    t.start()
    deadline = time.monotonic() + 2
    while (
        monitor.store.recent_events()[0]["sms_status"] == "pending" and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    monitor._stop.set()
    t.join(2)
    assert monitor.store.recent_events()[0]["sms_status"] == "dry-run"


def test_capture_read_failure(monitor, monkeypatch):
    c = Mock()
    c.isOpened.return_value = True
    c.read.return_value = (False, None)
    monkeypatch.setattr("app.monitor.cv2.VideoCapture", lambda _: c)
    monitor._capture_loop()
    assert (
        monitor.status()["state"] == "camera-error"
        and "stopped returning" in monitor.status()["error"]
    )
    c.release.assert_called_once()


def test_configuration_env(monkeypatch):
    monkeypatch.setenv("DETECTION_STRIDE", "0")
    monkeypatch.setenv("AUTO_START", "yes")
    c = AppConfig.from_env()
    assert c.detection_stride == 1 and c.auto_start is True


def test_overlay_zones():
    f = np.zeros((240, 320, 3), np.uint8)
    z = Zone("zone", ((0.1, 0.1), (0.9, 0.1), (0.5, 0.9)))
    assert (
        draw_overlay(
            f, VisionResult([], 0, []), ThreatAssessment(0, "low", "none", (), False), [z]
        ).shape
        == f.shape
    )


def test_empty_zone_rejected():
    with pytest.raises(ValueError):
        Zone("bad", ((0, 0), (1, 1)))


def test_nonperson_score_and_quiet_summary():
    a = ThreatEvaluator().evaluate(
        [Detection("car", 0.9, (1, 1, 2, 2))], 0, [], (100, 100), MonitorRules(), datetime.now()
    )
    assert a.score == 6
    a = ThreatEvaluator().evaluate([], 0, [], (100, 100), MonitorRules(), datetime.now())
    assert a.summary == "No notable activity"


def test_inference_recovers_after_transient_failure():
    detector = ObjectDetector("fake", 0.5)
    detector._model = Mock()
    detector._model.predict.side_effect = [RuntimeError("temporary"), []]
    frame = np.zeros((32, 32, 3), np.uint8)
    assert detector.detect(frame) == []
    assert detector.error == "temporary"
    assert detector.detect(frame) == []
    assert detector.error is None


def test_snapshot_directory_failure_keeps_event(monitor, monkeypatch):
    from pathlib import Path

    def fail(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr(Path, "mkdir", fail)
    p = packet()
    monitor._record_event(
        p, ThreatAssessment(80, "critical", "test", (), False), p.source.frame, 100
    )
    assert monitor.store.recent_events()[0]["snapshot"] is None


@pytest.mark.parametrize("points", [((0, 0), (0.5, 0.5), (1, 1)), ((0.5, 0.5),) * 3])
def test_zero_area_domain_zone_rejected(points):
    with pytest.raises(ValueError, match="nonzero area"):
        Zone("invalid", points)
