from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, time


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    points: tuple[tuple[float, float], ...]
    enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("A zone needs at least three points")
        for x, y in self.points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("Zone points must be normalized between 0 and 1")

    def as_dict(self) -> dict:
        return {"name": self.name, "points": [list(point) for point in self.points], "enabled": self.enabled}


@dataclass(frozen=True, slots=True)
class MonitorRules:
    schedule_enabled: bool = False
    active_from: str = "22:00"
    active_until: str = "06:00"
    watch_labels: tuple[str, ...] = ("person", "car", "truck", "dog")

    def is_after_hours(self, moment: datetime) -> bool:
        if not self.schedule_enabled:
            return False
        start = time.fromisoformat(self.active_from)
        end = time.fromisoformat(self.active_until)
        current = moment.time().replace(tzinfo=None)
        if start <= end:
            return not (start <= current <= end)
        return end < current < start


@dataclass(frozen=True, slots=True)
class ThreatAssessment:
    score: int
    level: str
    summary: str
    inside_zones: tuple[str, ...]
    after_hours: bool


def point_in_polygon(point: tuple[float, float], polygon: Iterable[tuple[float, float]]) -> bool:
    """Ray-casting polygon test kept independent from OpenCV for easy testing."""
    x, y = point
    vertices = list(polygon)
    inside = False
    j = len(vertices) - 1
    for i, (xi, yi) in enumerate(vertices):
        xj, yj = vertices[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            boundary_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < boundary_x:
                inside = not inside
        j = i
    return inside


class ThreatEvaluator:
    def evaluate(
        self,
        detections: list[Detection],
        motion_ratio: float,
        zones: list[Zone],
        frame_size: tuple[int, int],
        rules: MonitorRules,
        observed_at: datetime,
    ) -> ThreatAssessment:
        width, height = frame_size
        watched = [item for item in detections if item.label in rules.watch_labels]
        people = [item for item in watched if item.label == "person"]

        score = min(30, round(max(0.0, motion_ratio) * 300))
        if people:
            score += 38 + min(12, 4 * (len(people) - 1))
        score += min(18, 6 * len([item for item in watched if item.label != "person"]))

        triggered_zones: set[str] = set()
        for detection in watched:
            x1, y1, x2, y2 = detection.box
            center = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height))
            for zone in zones:
                if zone.enabled and point_in_polygon(center, zone.points):
                    triggered_zones.add(zone.name)
        if triggered_zones:
            score += 25

        after_hours = rules.is_after_hours(observed_at)
        if after_hours and watched:
            score += 15
        score = min(100, score)

        if score >= 75:
            level = "critical"
        elif score >= 55:
            level = "high"
        elif score >= 30:
            level = "medium"
        else:
            level = "low"

        labels = sorted({item.label for item in watched})
        if not labels and motion_ratio > 0:
            summary = "Motion detected"
        elif labels:
            summary = f"Detected {', '.join(labels)}"
        else:
            summary = "No notable activity"
        if triggered_zones:
            summary += f" in {', '.join(sorted(triggered_zones))}"
        if after_hours and watched:
            summary += " outside the active schedule"

        return ThreatAssessment(score, level, summary, tuple(sorted(triggered_zones)), after_hours)

