from __future__ import annotations

import os


def telegram_disabled() -> bool:
    """Global kill switch for Telegram notifications.

    Telegram is disabled by default while the bot is under research/testing.
    Set TELEGRAM_DISABLED=0 only when notifications should be explicitly
    re-enabled.
    """
    value = os.getenv("TELEGRAM_DISABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def print_telegram_disabled() -> None:
    print("Telegram notifications are disabled by TELEGRAM_DISABLED kill switch.")
