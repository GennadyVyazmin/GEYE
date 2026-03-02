import threading
from datetime import datetime, timedelta, timezone

from .db import db_conn


class AnalyticsService:
    def __init__(
        self,
        db_path: str,
        session_id: str,
        min_db_event_interval_sec: float,
        count_confirm_min_hits: int,
        count_confirm_min_age_sec: float,
        online_ttl_sec: float,
        enable_line_crossing: bool,
        line_y_ratio: float,
        crossing_debounce_sec: float,
    ) -> None:
        self.db_path = db_path
        self.session_id = session_id
        self.min_db_event_interval = timedelta(seconds=min_db_event_interval_sec)
        self.count_confirm_min_hits = max(1, count_confirm_min_hits)
        self.count_confirm_min_age = timedelta(seconds=max(0.0, count_confirm_min_age_sec))
        self.online_ttl = timedelta(seconds=online_ttl_sec)
        self.enable_line_crossing = enable_line_crossing
        self.crossing_debounce = timedelta(seconds=crossing_debounce_sec)
        self.line_y_ratio = max(0.05, min(0.95, line_y_ratio))
        self._lock = threading.Lock()
        self._last_db_write_by_global: dict[int, datetime] = {}
        self._last_seen_by_global: dict[int, datetime] = {}
        self._first_seen_by_global: dict[int, datetime] = {}
        self._seen_hits_by_global: dict[int, int] = {}
        self._confirmed_globals: set[int] = set()
        self._last_center_y_norm_by_global: dict[int, float] = {}
        self._last_crossing_by_global: dict[int, datetime] = {}

    def reset_state(self, clear_db: bool = True) -> None:
        with self._lock:
            self._last_db_write_by_global.clear()
            self._last_seen_by_global.clear()
            self._first_seen_by_global.clear()
            self._seen_hits_by_global.clear()
            self._confirmed_globals.clear()
            self._last_center_y_norm_by_global.clear()
            self._last_crossing_by_global.clear()

        if not clear_db:
            return
        with db_conn(self.db_path) as conn:
            conn.execute("DELETE FROM sightings")
            conn.execute("DELETE FROM crossings")
            conn.commit()

    def register_seen(self, global_id: int, now: datetime) -> None:
        should_write = False
        with self._lock:
            self._last_seen_by_global[global_id] = now
            self._seen_hits_by_global[global_id] = self._seen_hits_by_global.get(global_id, 0) + 1
            first_seen = self._first_seen_by_global.get(global_id)
            if first_seen is None:
                first_seen = now
                self._first_seen_by_global[global_id] = now

            is_confirmed = global_id in self._confirmed_globals
            if not is_confirmed:
                enough_hits = self._seen_hits_by_global[global_id] >= self.count_confirm_min_hits
                enough_age = (now - first_seen) >= self.count_confirm_min_age
                if enough_hits and enough_age:
                    self._confirmed_globals.add(global_id)
                else:
                    return

            prev = self._last_db_write_by_global.get(global_id)
            if prev is not None and (now - prev) < self.min_db_event_interval:
                return
            self._last_db_write_by_global[global_id] = now
            should_write = True

        if not should_write:
            return
        with db_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sightings (ts, session_id, global_id)
                VALUES (?, ?, ?)
                """,
                (now.isoformat(), self.session_id, global_id),
            )
            conn.commit()

    def register_position(
        self, global_id: int, center_y_px: float, frame_height_px: int, now: datetime
    ) -> str | None:
        if not self.enable_line_crossing:
            return None
        if frame_height_px <= 0:
            return None
        center_norm = center_y_px / float(frame_height_px)
        direction: str | None = None

        with self._lock:
            prev = self._last_center_y_norm_by_global.get(global_id)
            self._last_center_y_norm_by_global[global_id] = center_norm
            if prev is None:
                return None

            crossed_in = prev < self.line_y_ratio <= center_norm
            crossed_out = prev > self.line_y_ratio >= center_norm
            if not crossed_in and not crossed_out:
                return None

            last_cross = self._last_crossing_by_global.get(global_id)
            if last_cross is not None and (now - last_cross) < self.crossing_debounce:
                return None

            self._last_crossing_by_global[global_id] = now
            direction = "in" if crossed_in else "out"

        with db_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO crossings (ts, session_id, global_id, direction)
                VALUES (?, ?, ?, ?)
                """,
                (now.isoformat(), self.session_id, global_id, direction),
            )
            conn.commit()
        return direction

    def get_stats(self) -> dict:
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)

        with db_conn(self.db_path) as conn:
            hour_row = conn.execute(
                """
                SELECT COUNT(DISTINCT session_id || '-' || global_id)
                FROM sightings
                WHERE ts >= ?
                """,
                (hour_ago.isoformat(),),
            ).fetchone()
            day_row = conn.execute(
                """
                SELECT COUNT(DISTINCT session_id || '-' || global_id)
                FROM sightings
                WHERE ts >= ?
                """,
                (day_ago.isoformat(),),
            ).fetchone()
            if self.enable_line_crossing:
                in_hour_row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM crossings
                    WHERE ts >= ? AND direction = 'in'
                    """,
                    (hour_ago.isoformat(),),
                ).fetchone()
                out_hour_row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM crossings
                    WHERE ts >= ? AND direction = 'out'
                    """,
                    (hour_ago.isoformat(),),
                ).fetchone()
                in_day_row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM crossings
                    WHERE ts >= ? AND direction = 'in'
                    """,
                    (day_ago.isoformat(),),
                ).fetchone()
                out_day_row = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM crossings
                    WHERE ts >= ? AND direction = 'out'
                    """,
                    (day_ago.isoformat(),),
                ).fetchone()
            else:
                in_hour_row = (0,)
                out_hour_row = (0,)
                in_day_row = (0,)
                out_day_row = (0,)

        online_ids: list[int] = []
        with self._lock:
            for global_id, seen_at in self._last_seen_by_global.items():
                if (now - seen_at) <= self.online_ttl:
                    online_ids.append(global_id)

        return {
            "timestamp_utc": now.isoformat(),
            "line_crossing_enabled": self.enable_line_crossing,
            "line_y_ratio": self.line_y_ratio,
            "online_count": len(online_ids),
            "online_global_ids": sorted(online_ids),
            "unique_last_hour": int(hour_row[0] or 0),
            "unique_last_24h": int(day_row[0] or 0),
            "entries_last_hour": int(in_hour_row[0] or 0),
            "exits_last_hour": int(out_hour_row[0] or 0),
            "entries_last_24h": int(in_day_row[0] or 0),
            "exits_last_24h": int(out_day_row[0] or 0),
        }
