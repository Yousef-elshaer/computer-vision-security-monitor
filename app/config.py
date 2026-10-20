from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera_index: int = 0
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = 0.45
    detection_stride: int = 3
    auto_start: bool = True
    event_threshold: int = 35
    alert_threshold: int = 70
    alert_cooldown_seconds: int = 90
    sms_dry_run: bool = True
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    alert_to_number: str = ""
    data_dir: Path = PROJECT_ROOT / "data"

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            camera_index=int(os.getenv("CAMERA_INDEX", "0")),
            yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt"),
            yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.45")),
            detection_stride=max(1, int(os.getenv("DETECTION_STRIDE", "3"))),
            auto_start=_as_bool("AUTO_START", True),
            event_threshold=int(os.getenv("EVENT_THRESHOLD", "35")),
            alert_threshold=int(os.getenv("ALERT_THRESHOLD", "70")),
            alert_cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "90")),
            sms_dry_run=_as_bool("SMS_DRY_RUN", True),
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_from_number=os.getenv("TWILIO_FROM_NUMBER", ""),
            alert_to_number=os.getenv("ALERT_TO_NUMBER", ""),
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "watchtower.db"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"

