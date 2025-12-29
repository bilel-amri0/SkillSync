"""Minimal email helper used by the interview agent.

Real SMTP delivery is optional—if credentials are missing we simply log the
payload so the API remains deterministic for local development.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self) -> None:
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("REPORTS_FROM_EMAIL", "reports@skillsync.ai")

    def queue_report_email(self, *, to_email: str, subject: str, report_payload: Dict[str, object]) -> None:
        """Send or log the report email depending on configuration."""
        if not to_email:
            logger.info("Skipping email send – recipient missing")
            return

        if not (self.smtp_host and self.smtp_username and self.smtp_password):
            logger.info(
                "Email not configured. Would send report to %s with subject '%s'. Payload preview: %s",
                to_email,
                subject,
                {k: report_payload.get(k) for k in ("overall_score", "summary", "decision")},
            )
            return

        message = EmailMessage()
        message["From"] = self.from_email
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(
            "\n".join(
                [
                    report_payload.get("summary", "Interview summary attached."),
                    "",
                    f"Overall Score: {report_payload.get('overall_score', 'N/A')}",
                    f"Decision: {report_payload.get('decision', 'Consider')}",
                ]
            )
        )

        self._send(message)

    def _send(self, message: EmailMessage) -> None:
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self.smtp_username, self.smtp_password)
                smtp.send_message(message)
                logger.info("Report email sent to %s", message["To"])
        except Exception as exc:
            logger.warning("Failed to send email: %s", exc)
