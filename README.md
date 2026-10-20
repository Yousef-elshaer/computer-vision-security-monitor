# Watchtower Vision

Watchtower Vision is a local webcam security monitor. It combines motion detection with YOLO object detection, scores each observation, records meaningful events, and can send high-risk alerts over SMS. A responsive FastAPI dashboard shows the annotated live feed, recent snapshots, monitoring status, schedules, and custom detection zones.

## What it includes

- A four-worker pipeline for camera capture, object detection, threat analysis, and SMS delivery.
- Person and object detection through Ultralytics YOLO, plus OpenCV motion detection.
- Custom polygon zones drawn directly on the live dashboard.
- Optional active hours and watched-object rules.
- Risk scoring from 0-100 with low, medium, high, and critical levels.
- SQLite event history, JPEG evidence snapshots, and JSON API endpoints.
- Twilio SMS alerts with cooldown protection and a safe dry-run default.
- Graceful camera/model error reporting: the dashboard stays available when hardware is missing.

## Quick start

Python 3.11 or newer is recommended. A webcam must be connected to the same computer that runs the server.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev,vision]"
cp .env.example .env
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The first object-detection run may download the configured YOLO weights. If camera index `0` is not the intended webcam, change `CAMERA_INDEX` in `.env` and restart the server.

## Enable SMS alerts

Create a Twilio account and an SMS-capable sender, then set these values in `.env`:

```dotenv
SMS_DRY_RUN=false
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
ALERT_TO_NUMBER=+1...
```

Use E.164 phone-number format, including the country code. Leave `SMS_DRY_RUN=true` until the dashboard's test-alert button returns `dry-run`, then switch it off and test with your real credentials. Trial Twilio accounts may require the recipient number to be verified.

## How risk scoring works

The scoring logic is intentionally easy to audit and change in `app/domain.py`:

- Motion contributes up to 30 points.
- A person contributes 38-50 points depending on the count.
- Other watched objects contribute up to 18 points.
- An object centered inside an enabled zone adds 25 points.
- Watched activity outside the configured schedule adds 15 points.

Events are recorded at `EVENT_THRESHOLD` (35 by default). SMS is considered at `ALERT_THRESHOLD` (70), then rate-limited by `ALERT_COOLDOWN_SECONDS`.

## Useful endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/healthz` | Lightweight health check |
| `GET` | `/video-feed` | Annotated MJPEG stream |
| `GET` | `/api/status` | Camera, model, risk, and SMS state |
| `GET` | `/api/events` | Recent security events |
| `POST` | `/api/monitor/arm` | Arm or disarm event recording |
| `GET/PUT` | `/api/zones` | Read or replace detection zones |
| `GET/PUT` | `/api/rules` | Read or update schedules and labels |
| `POST` | `/api/alerts/test` | Test the configured SMS mode |

Interactive API documentation is available at `/docs` while the app is running.

## Tests and code quality

```bash
pytest
ruff check app tests
```

The unit tests do not require a webcam, YOLO weights, or Twilio account. They cover threat scoring, polygon checks, schedules, persistence, and SMS safety behavior.

If you only need the dashboard, motion detection, and tests on a development machine, install `.[dev]` without the `vision` extra. This avoids installing PyTorch until object detection is needed.

## Practical notes

- Keep this service on a trusted local network. Add authentication and HTTPS before exposing it outside your home network.
- The default model is small enough for many laptops, but CPU inference speed varies. Increase `DETECTION_STRIDE` if the live view lags.
- Snapshots and SQLite files live under `data/` and are intentionally ignored by Git.
- Twilio credentials belong only in `.env`; never commit that file.
