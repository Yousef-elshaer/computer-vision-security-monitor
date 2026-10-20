from app.database import EventStore


def test_event_round_trip_and_stats(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.initialize()
    event_id = store.add_event(
        created_at="2026-10-19T18:00:00+00:00",
        level="high",
        score=72,
        summary="Detected person in Front door",
        snapshot="/snapshots/test.jpg",
        detections=[{"label": "person", "confidence": 0.9, "box": [1, 2, 3, 4]}],
        sms_status="pending",
    )
    store.update_sms_status(event_id, "dry-run")
    event = store.recent_events(1)[0]
    assert event["sms_status"] == "dry-run"
    assert event["detections"][0]["label"] == "person"
    assert store.stats()["high_risk"] == 1


def test_settings_are_json_persisted(tmp_path):
    store = EventStore(tmp_path / "settings.db")
    store.initialize()
    store.set_setting("armed", False)
    assert store.get_setting("armed", True) is False

