from datetime import datetime

import pytest

from app.domain import Detection, MonitorRules, ThreatEvaluator, Zone, point_in_polygon


def test_point_in_polygon_handles_inside_and_outside_points():
    square = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((0.95, 0.5), square)


def test_zone_rejects_unusable_coordinates():
    with pytest.raises(ValueError):
        Zone("broken", ((0, 0), (1.2, 0), (0, 1)))


def test_person_inside_zone_reaches_high_risk():
    evaluator = ThreatEvaluator()
    assessment = evaluator.evaluate(
        detections=[Detection("person", 0.91, (400, 200, 600, 800))],
        motion_ratio=0.03,
        zones=[Zone("Front door", ((0.3, 0.1), (0.8, 0.1), (0.8, 0.9), (0.3, 0.9)))],
        frame_size=(1000, 1000),
        rules=MonitorRules(),
        observed_at=datetime(2026, 1, 10, 12, 0),
    )
    assert assessment.level == "high"
    assert assessment.inside_zones == ("Front door",)
    assert "Front door" in assessment.summary


def test_overnight_schedule_flags_midday_activity():
    rules = MonitorRules(schedule_enabled=True, active_from="22:00", active_until="06:00")
    assert rules.is_after_hours(datetime(2026, 1, 10, 12, 0))
    assert not rules.is_after_hours(datetime(2026, 1, 10, 23, 0))

