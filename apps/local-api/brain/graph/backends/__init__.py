"""Backend selection: which model answers this turn, and why.

`route_backend` needs an answer before it calls anything, but two of the three
fallback conditions -- the network is down, today's free quota is spent -- are
only knowable by having tried. `CloudAvailability` is the memory that closes
that gap: a failed call parks the cloud route for a while, and the router reads
the park rather than rediscovering the failure every turn.

The two parks have very different lengths on purpose. A network blip clears in
about a minute; a spent daily quota does not clear until Google's window rolls
over. Using one duration for both either strands the conversation on a 1.7B all
day over a dropped packet, or spends a turn's latency every minute re-proving
that the quota is still gone.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from brain.graph.backends.base import (
    BackendError,
    BackendUnavailable,
    ChatBackend,
    DailyQuotaExceeded,
    Generation,
    GenerationOptions,
    Usage,
)
from brain.graph.backends.cloud import CloudBackend
from brain.graph.backends.local import LocalBackend

__all__ = [
    "BackendError",
    "BackendRegistry",
    "BackendUnavailable",
    "ChatBackend",
    "CloudAvailability",
    "CloudBackend",
    "DailyQuotaExceeded",
    "Generation",
    "GenerationOptions",
    "LocalBackend",
    "Usage",
]

logger = logging.getLogger(__name__)


class CloudAvailability:
    """Remembers why the cloud route last failed, and until when."""

    def __init__(
        self,
        transient_cooldown_seconds: float = 60.0,
        quota_reset_timezone: str = "America/Los_Angeles",
    ) -> None:
        self._transient_cooldown = transient_cooldown_seconds
        # Google AI Studio's free-tier daily quotas roll over at midnight
        # Pacific, not at the caller's local midnight and not 24h after the
        # failure -- parking for a fixed 24h would skip most of a free day.
        self._quota_tz = ZoneInfo(quota_reset_timezone)
        self._parked_until: float = 0.0
        self._reason: str = ""

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _seconds_until_quota_reset(self) -> float:
        now = datetime.now(self._quota_tz)
        reset = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return max((reset - now).total_seconds(), 60.0)

    def record_success(self) -> None:
        if self._parked_until:
            logger.info("cloud route recovered")
        self._parked_until = 0.0
        self._reason = ""

    def record_transient(self, detail: str) -> None:
        self._park(self._transient_cooldown, f"unavailable: {detail}")

    def record_daily_quota(self, detail: str) -> None:
        self._park(self._seconds_until_quota_reset(), f"daily_quota: {detail}")

    def _park(self, seconds: float, reason: str) -> None:
        self._parked_until = self._now() + seconds
        self._reason = reason
        logger.warning("cloud route parked for %.0fs | %s", seconds, reason)

    @property
    def reason(self) -> str:
        """Why the cloud route is parked, or "" when it is open."""
        return self._reason if not self.is_available else ""

    @property
    def is_available(self) -> bool:
        if not self._parked_until:
            return True
        if self._now() >= self._parked_until:
            self._parked_until = 0.0
            self._reason = ""
            return True
        return False


class BackendRegistry:
    """The backends this process can reach, and the rule for picking one."""

    def __init__(
        self,
        local: LocalBackend,
        cloud: CloudBackend | None,
        availability: CloudAvailability,
        prefer_cloud: bool = True,
    ) -> None:
        self.local = local
        self.cloud = cloud
        self.availability = availability
        self.prefer_cloud = prefer_cloud

    def get(self, route: str) -> ChatBackend:
        if route == "cloud" and self.cloud is not None:
            return self.cloud
        return self.local

    def choose(self) -> tuple[str, str]:
        """Return (route, reason). Reason is "" when the default route stands."""
        if not self.prefer_cloud:
            return "local", "cloud_disabled"
        if self.cloud is None:
            # Condition (c): no key configured, so there is nothing to try.
            return "local", "no_api_key"
        if not self.availability.is_available:
            # Conditions (a) and (b), remembered from the call that hit them.
            return "local", self.availability.reason or "cloud_parked"
        return "cloud", ""
