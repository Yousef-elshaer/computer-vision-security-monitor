from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig


@dataclass(frozen=True, slots=True)
class SendResult:
    status: str
    message_id: str | None = None


class SMSNotifier:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def configured(self) -> bool:
        return all(
            (
                self.config.twilio_account_sid,
                self.config.twilio_auth_token,
                self.config.twilio_from_number,
                self.config.alert_to_number,
            )
        )

    def send(self, body: str) -> SendResult:
        if self.config.sms_dry_run:
            return SendResult("dry-run")
        if not self.configured:
            return SendResult("not-configured")

        from twilio.rest import Client

        client = Client(self.config.twilio_account_sid, self.config.twilio_auth_token)
        message = client.messages.create(
            body=body,
            from_=self.config.twilio_from_number,
            to=self.config.alert_to_number,
        )
        return SendResult("queued", message.sid)

