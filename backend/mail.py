"""Outbound email, with a delivery mechanism that does not require a mail provider.

Password reset is a required public route, and a classroom demonstration should not need an
SMTP account to have one. So delivery is an interface with two implementations:

* :class:`ConsoleMailer` — prints the message to the server log. The default, and what runs
  in development. The reset link is visible in the uvicorn output.
* :class:`SmtpMailer` — used automatically as soon as ``SMTP_HOST`` is configured.

The reset flow itself is identical either way: the same single-use, expiring, hashed token.
Only the last inch differs, which is the point — nothing about the security of the flow
depends on which mailer is active.
"""
from __future__ import annotations

import smtplib
from email.message import EmailMessage

from backend.settings import settings


class Mailer:
    def send(self, to: str, subject: str, body: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ConsoleMailer(Mailer):
    """Logs the message. Never used when SMTP is configured."""

    def send(self, to: str, subject: str, body: str) -> None:
        print(
            "\n"
            "──────────────────────── EMAIL (not sent) ────────────────────────\n"
            f"To:      {to}\n"
            f"Subject: {subject}\n"
            "──────────────────────────────────────────────────────────────────\n"
            f"{body}\n"
            "──────────────────────────────────────────────────────────────────\n",
            flush=True,
        )


class SmtpMailer(Mailer):
    def send(self, to: str, subject: str, body: str) -> None:  # pragma: no cover - needs a server
        message = EmailMessage()
        message["From"] = settings.smtp_from
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
            if settings.smtp_starttls:
                server.starttls()
            if settings.smtp_user and settings.smtp_password:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)


def get_mailer() -> Mailer:
    return SmtpMailer() if settings.smtp_configured else ConsoleMailer()


def reset_email(reset_url: str, ttl_minutes: int) -> tuple[str, str]:
    """``(subject, body)`` for a password-reset message."""
    subject = "Reset your Multi-Disease AI password"
    body = (
        "Someone asked to reset the password for this address.\n\n"
        f"{reset_url}\n\n"
        f"The link works once and expires in {ttl_minutes} minutes.\n\n"
        "If it was not you, nothing has changed and you can ignore this message. "
        "Your password stays the same until the link is used.\n"
    )
    return subject, body
