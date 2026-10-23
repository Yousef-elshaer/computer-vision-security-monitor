from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from app.config import PROJECT_ROOT, AppConfig
from app.database import EventStore
from app.domain import MonitorRules, Zone
from app.monitor import SecurityMonitor

config = AppConfig.from_env()
store = EventStore(config.database_path)
store.initialize()
monitor = SecurityMonitor(config, store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if config.auto_start:
        monitor.start()
    yield
    monitor.stop()


app = FastAPI(title="Watchtower Vision", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")
app.mount("/snapshots", StaticFiles(directory=config.snapshot_dir, check_dir=False), name="snapshots")
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")


class ArmRequest(BaseModel):
    armed: bool


class ZonePayload(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=20)
    enabled: bool = True

    @field_validator("points")
    @classmethod
    def normalized_points(cls, points: list[tuple[float, float]]):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
            raise ValueError("Every point must be normalized between 0 and 1")
        Zone("validation", tuple(points))
        return points


class ZoneListPayload(BaseModel):
    zones: list[ZonePayload] = Field(max_length=8)


class RulePayload(BaseModel):
    schedule_enabled: bool = False
    active_from: str = "22:00"
    active_until: str = "06:00"
    watch_labels: list[str] = Field(default_factory=lambda: ["person", "car", "truck", "dog"], min_length=1)

    @field_validator("active_from", "active_until")
    @classmethod
    def valid_time(cls, value: str):
        try:
            hour, minute = (int(part) for part in value.split(":"))
        except ValueError as exc:
            raise ValueError("Use a 24-hour HH:MM time") from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Use a 24-hour HH:MM time")
        return f"{hour:02d}:{minute:02d}"


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/healthz")
async def health():
    return {"ok": True, "monitor": monitor.status()["state"]}


@app.get("/video-feed")
async def video_feed():
    return StreamingResponse(monitor.jpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
async def get_status():
    return monitor.status()


@app.post("/api/monitor/arm")
async def set_armed(payload: ArmRequest):
    monitor.set_armed(payload.armed)
    return {"armed": monitor.armed}


@app.get("/api/events")
async def get_events(limit: int = Query(default=30, ge=1, le=200)):
    return {"events": store.recent_events(limit)}


@app.get("/api/stats")
async def get_stats():
    return store.stats()


@app.get("/api/zones")
async def get_zones():
    return {"zones": [zone.as_dict() for zone in monitor.zones]}


@app.put("/api/zones")
async def update_zones(payload: ZoneListPayload):
    zones = [Zone(item.name, tuple(item.points), item.enabled) for item in payload.zones]
    monitor.set_zones(zones)
    return {"zones": [zone.as_dict() for zone in zones]}


@app.get("/api/rules")
async def get_rules():
    rules = monitor.rules
    return {
        "schedule_enabled": rules.schedule_enabled,
        "active_from": rules.active_from,
        "active_until": rules.active_until,
        "watch_labels": list(rules.watch_labels),
    }


@app.put("/api/rules")
async def update_rules(payload: RulePayload):
    rules = MonitorRules(
        schedule_enabled=payload.schedule_enabled,
        active_from=payload.active_from,
        active_until=payload.active_until,
        watch_labels=tuple(dict.fromkeys(label.strip().lower() for label in payload.watch_labels if label.strip())),
    )
    if not rules.watch_labels:
        raise HTTPException(422, "At least one watched label is required")
    monitor.set_rules(rules)
    return await get_rules()


@app.post("/api/alerts/test")
async def test_alert():
    try:
        return {"status": monitor.queue_test_alert()}
    except Exception as exc:
        raise HTTPException(502, f"SMS provider error: {type(exc).__name__}") from exc


