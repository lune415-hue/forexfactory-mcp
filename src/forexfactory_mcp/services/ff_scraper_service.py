import asyncio
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import certifi
import httpx

from forexfactory_mcp.models.time_period import TimePeriod
from forexfactory_mcp.settings import get_settings

logger = logging.getLogger(__name__)

# ForexFactory's official JSON calendar export, published by faireconomy.
# This is the SAME data the calendar HTML page renders, but served as a plain
# JSON document with NO Cloudflare anti-bot challenge — so it does not require
# Playwright/a headless browser and cannot be silently blocked.
#
# WHY THIS REPLACED THE PLAYWRIGHT SCRAPE (2026-05-27):
#   ForexFactory turned on a Cloudflare "Just a moment…" bot-verification
#   interstitial on /calendar. Headless Chromium never clears the challenge, so
#   window.calendarComponentStates never loads, page.wait_for_function timed out
#   after 45s, and ff_scraper_service swallowed it into an empty list — the tool
#   returned [] for EVERY period, masquerading as "no events". The 2026-05-25
#   timeout bump was correct for the *old* failure (slow load) but is moot
#   against a Cloudflare wall. The faireconomy JSON feed sidesteps it entirely.
#
# Feed coverage: faireconomy currently publishes ONLY the "thisweek" file
# (nextweek/lastweek/month variants now 404). It contains every event for the
# current ForexFactory week (Sun–Sat, US-Eastern), which covers today /
# tomorrow / yesterday / this_week / in-week custom ranges. Periods outside the
# current week (next_week, last_week, this_month, …) cannot be served from this
# feed and return an empty result with a logged warning.
FF_JSON_THISWEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# ForexFactory assigns calendar days in US-Eastern; filter day boundaries there
# so "today" matches what the FF calendar shows.
FF_TZ = ZoneInfo("America/New_York")

# faireconomy rate-limits rapid repeated hits (429 Too Many Requests). The feed
# is a weekly event schedule, not a price — cache the payload process-wide for a
# few minutes so multiple period queries in one session share a single fetch.
_FEED_CACHE_TTL_S = 600  # 10 min
_feed_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}


class FFScraperService:
    """
    Fetches the ForexFactory economic calendar via the official faireconomy
    JSON export (no Playwright, no Cloudflare wall).

    The public interface is unchanged from the previous Playwright-based
    implementation: construct with a ``TimePeriod`` (or ``TimePeriod.CUSTOM``
    plus start/end dates) and ``await get_events()``. ``get_events()`` returns a
    list of "day blocks" (``[{"events": [...]}]``) so the downstream
    ``extract_and_normalize_events`` pipeline keeps working byte-for-byte.

    Attributes
    ----------
    settings : object
        Project settings (timeouts, headers, timezone).
    time_period : TimePeriod
        Enum value specifying which calendar period to fetch.
    custom_start_date : Optional[str]
        Start date string (YYYY-MM-DD) if using TimePeriod.CUSTOM.
    custom_end_date : Optional[str]
        End date string (YYYY-MM-DD) if using TimePeriod.CUSTOM.
    """

    def __init__(
        self,
        time_period: TimePeriod,
        custom_start_date: Optional[str] = None,
        custom_end_date: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.time_period = time_period
        self.custom_start_date = custom_start_date
        self.custom_end_date = custom_end_date

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    async def get_events(self) -> List[Dict[str, Any]]:
        """
        Fetch the calendar and return events as day blocks.

        Returns
        -------
        List[Dict[str, Any]]
            ``[{"events": [<raw event>, ...]}]`` — a single day block whose
            ``events`` list is shaped for ``_normalize_event`` (``country`` →
            currency fallback, ``dateline`` unix ts → datetime enrichment).
            Empty ``events`` for an out-of-feed period or on fetch failure.
        """
        raw_feed = await self._fetch_thisweek()
        if raw_feed is None:
            # Fetch/parse failed — surface as empty (callers treat empty as
            # UNKNOWN, never a confident "no events").
            return [{"events": []}]

        start, end = self._date_range()
        if start is None or end is None:
            logger.warning(
                "⚠️ TimePeriod %s is outside the faireconomy thisweek feed "
                "coverage; returning no events.",
                self.time_period,
            )
            return [{"events": []}]

        events: List[Dict[str, Any]] = []
        for ev in raw_feed:
            ev_date = self._event_eastern_date(ev.get("date"))
            if ev_date is None or not (start <= ev_date <= end):
                continue
            events.append(self._to_raw_event(ev))

        logger.info(
            "✅ faireconomy feed: %d events in range %s..%s (period=%s)",
            len(events),
            start,
            end,
            self.time_period.value,
        )
        return [{"events": events}]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    async def _fetch_thisweek(self) -> Optional[List[Dict[str, Any]]]:
        """Return the thisweek JSON feed (cached), or None on error.

        Serves a process-wide cached payload within the TTL to avoid the
        faireconomy 429 rate limit; on a cache miss, fetches with one retry on a
        transient 429/5xx before giving up (→ None → empty result upstream).
        """
        now = time.monotonic()
        cached = _feed_cache.get(FF_JSON_THISWEEK_URL)
        if cached and (now - cached[0]) < _FEED_CACHE_TTL_S:
            logger.info("🗄  Using cached FF feed (%ds old)", int(now - cached[0]))
            return cached[1]

        timeout_s = max(5.0, self.settings.SCRAPER_TIMEOUT_MS / 1000.0)
        headers = {
            "User-Agent": self.settings.USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": self.settings.ACCEPT_LANGUAGE,
        }
        logger.info("🌐 Fetching ForexFactory JSON feed: %s", FF_JSON_THISWEEK_URL)
        # Retry only transient SERVER/network errors. A 429 means "back off" —
        # retrying it immediately just adds load and lengthens the penalty, so we
        # do NOT retry a 429; we fall through to stale-cache/None.
        for attempt in (1, 2):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout_s, verify=certifi.where(), follow_redirects=True
                ) as client:
                    resp = await client.get(FF_JSON_THISWEEK_URL, headers=headers)
                    if resp.status_code == 429:
                        logger.warning(
                            "⚠️ FF feed rate-limited (429); not retrying. "
                            "Will serve stale cache if available."
                        )
                        break
                    if resp.status_code in (500, 502, 503, 504) and attempt == 1:
                        logger.warning(
                            "⚠️ FF feed %s on attempt %d; retrying after backoff",
                            resp.status_code,
                            attempt,
                        )
                        await asyncio.sleep(2.0)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                if not isinstance(data, list):
                    logger.error("⚠️ FF feed returned non-list payload: %s", type(data))
                    break
                _feed_cache[FF_JSON_THISWEEK_URL] = (time.monotonic(), data)
                return data
            except Exception as e:  # noqa: BLE001 — fail soft to empty, never raise
                if attempt == 1:
                    logger.warning("⚠️ FF feed fetch error %s; retrying", e)
                    await asyncio.sleep(2.0)
                    continue
                logger.exception("⚠️ Could not fetch ForexFactory JSON feed: %s", e)

        # Last-ditch: serve a stale cached payload if we have one.
        if cached:
            logger.warning("⚠️ Serving STALE cached FF feed after fetch failure")
            return cached[1]
        return None

    @staticmethod
    def _event_eastern_date(date_str: Optional[str]) -> Optional[date]:
        """Parse a feed ISO datetime (e.g. '2026-05-27T08:30:00-04:00') to the
        US-Eastern calendar date FF files it under."""
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(FF_TZ)
            return dt.date()
        except Exception:
            return None

    def _to_raw_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """Map a faireconomy event into the raw shape `_normalize_event` expects.

        Sets ``country`` (→ currency fallback) and ``dateline`` to a unix
        timestamp (→ datetime fallback, which the normalizer enriches into
        datetime_utc / datetime_local).
        """
        dateline: Any = ""
        date_str = ev.get("date")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                dateline = int(dt.timestamp())
            except Exception:
                dateline = ""
        return {
            "title": ev.get("title") or "",
            "country": ev.get("country") or "",
            "impact": ev.get("impact") or "",
            "forecast": ev.get("forecast") or None,
            "previous": ev.get("previous") or None,
            "actual": ev.get("actual") or None,
            "dateline": dateline,
        }

    def _date_range(self) -> tuple[Optional[date], Optional[date]]:
        """Resolve the inclusive [start, end] Eastern-date filter for the period.

        Returns (None, None) for periods the thisweek feed cannot cover
        (next_week, last_week, months, tomorrow/yesterday that fall outside the
        current FF week)."""
        today = datetime.now(FF_TZ).date()
        # FF week runs Sunday..Saturday.
        week_start = today - timedelta(days=(today.weekday() + 1) % 7)  # back to Sun
        week_end = week_start + timedelta(days=6)

        def _in_week(d: date) -> bool:
            return week_start <= d <= week_end

        tp = self.time_period
        if tp == TimePeriod.TODAY:
            return today, today
        if tp == TimePeriod.TOMORROW:
            d = today + timedelta(days=1)
            return (d, d) if _in_week(d) else (None, None)
        if tp == TimePeriod.YESTERDAY:
            d = today - timedelta(days=1)
            return (d, d) if _in_week(d) else (None, None)
        if tp == TimePeriod.THIS_WEEK:
            return week_start, week_end
        if tp == TimePeriod.CUSTOM and self.custom_start_date and self.custom_end_date:
            try:
                s = datetime.strptime(self.custom_start_date, "%Y-%m-%d").date()
                e = datetime.strptime(self.custom_end_date, "%Y-%m-%d").date()
            except Exception:
                return None, None
            # Clamp to feed coverage; if no overlap, nothing to serve.
            s = max(s, week_start)
            e = min(e, week_end)
            return (s, e) if s <= e else (None, None)
        # next_week / last_week / this_month / next_month / last_month: not in feed.
        return None, None
