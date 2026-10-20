from app.config import AppConfig
from app.sms import SMSNotifier


def test_sms_defaults_to_safe_dry_run():
    notifier = SMSNotifier(AppConfig(sms_dry_run=True))
    result = notifier.send("test")
    assert result.status == "dry-run"


def test_unconfigured_live_sms_is_reported_without_calling_provider():
    notifier = SMSNotifier(AppConfig(sms_dry_run=False))
    result = notifier.send("test")
    assert result.status == "not-configured"

