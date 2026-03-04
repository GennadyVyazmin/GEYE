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
        unique_require_face_for_count: bool,
        face_confirm_min_hits: int,
        count_confirm_no_face_fallback_enabled: bool,
        count_confirm_no_face_age_sec: float,
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
        self.unique_require_face_for_count = unique_require_face_for_count
        self.face_confirm_min_hits = max(1, face_confirm_min_hits)
        self.count_confirm_no_face_fallback_enabled = bool(count_confirm_no_face_fallback_enabled)
        self.count_confirm_no_face_age = timedelta(seconds=max(0.0, count_confirm_no_face_age_sec))
        self.online_ttl = timedelta(seconds=online_ttl_sec)
        self.enable_line_crossing = enable_line_crossing
        self.crossing_debounce = timedelta(seconds=crossing_debounce_sec)
        self.line_y_ratio = max(0.05, min(0.95, line_y_ratio))
        self._lock = threading.Lock()
        self._last_db_write_by_global: dict[int, datetime] = {}
        self._last_seen_by_global: dict[int, datetime] = {}
        self._first_seen_by_global: dict[int, datetime] = {}
        self._seen_hits_by_global: dict[int, int] = {}
        self._face_hits_by_global: dict[int, int] = {}
        self._confirmed_globals: set[int] = set()
        self._last_center_y_norm_by_global: dict[int, float] = {}
        self._last_crossing_by_global: dict[int, datetime] = {}

    def get_confirm_min_age_sec(self) -> float:
        with self._lock:
            return self.count_confirm_min_age.total_seconds()

    def set_confirm_min_age_sec(self, value: float) -> float:
        value = max(0.0, min(20.0, float(value)))
        with self._lock:
            self.count_confirm_min_age = timedelta(seconds=value)
            return self.count_confirm_min_age.total_seconds()

    def get_no_face_fallback_enabled(self) -> bool:
        with self._lock:
            return self.count_confirm_no_face_fallback_enabled

    def set_no_face_fallback_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self.count_confirm_no_face_fallback_enabled = bool(enabled)
            return self.count_confirm_no_face_fallback_enabled

    def get_tuning(self) -> dict:
        with self._lock:
            return {
                "count_confirm_min_hits": int(self.count_confirm_min_hits),
                "count_confirm_min_age_sec": float(self.count_confirm_min_age.total_seconds()),
                "unique_require_face_for_count": bool(self.unique_require_face_for_count),
                "face_confirm_min_hits": int(self.face_confirm_min_hits),
                "count_confirm_no_face_fallback_enabled": bool(self.count_confirm_no_face_fallback_enabled),
                "count_confirm_no_face_age_sec": float(self.count_confirm_no_face_age.total_seconds()),
                "min_db_event_interval_sec": float(self.min_db_event_interval.total_seconds()),
                "online_ttl_sec": float(self.online_ttl.total_seconds()),
            }

    def update_tuning(self, values: dict) -> dict:
        with self._lock:
            if "count_confirm_min_hits" in values:
                self.count_confirm_min_hits = max(1, int(values["count_confirm_min_hits"]))
            if "count_confirm_min_age_sec" in values:
                self.count_confirm_min_age = timedelta(seconds=max(0.0, float(values["count_confirm_min_age_sec"])))
            if "unique_require_face_for_count" in values:
                self.unique_require_face_for_count = bool(values["unique_require_face_for_count"])
            if "face_confirm_min_hits" in values:
                self.face_confirm_min_hits = max(1, int(values["face_confirm_min_hits"]))
            if "count_confirm_no_face_fallback_enabled" in values:
                self.count_confirm_no_face_fallback_enabled = bool(values["count_confirm_no_face_fallback_enabled"])
            if "count_confirm_no_face_age_sec" in values:
                self.count_confirm_no_face_age = timedelta(seconds=max(0.0, float(values["count_confirm_no_face_age_sec"])))
            if "min_db_event_interval_sec" in values:
                self.min_db_event_interval = timedelta(seconds=max(0.1, float(values["min_db_event_interval_sec"])))
            if "online_ttl_sec" in values:
                self.online_ttl = timedelta(seconds=max(0.1, float(values["online_ttl_sec"])))
            return {
                "count_confirm_min_hits": int(self.count_confirm_min_hits),
                "count_confirm_min_age_sec": float(self.count_confirm_min_age.total_seconds()),
                "unique_require_face_for_count": bool(self.unique_require_face_for_count),
                "face_confirm_min_hits": int(self.face_confirm_min_hits),
                "count_confirm_no_face_fallback_enabled": bool(self.count_confirm_no_face_fallback_enabled),
                "count_confirm_no_face_age_sec": float(self.count_confirm_no_face_age.total_seconds()),
                "min_db_event_interval_sec": float(self.min_db_event_interval.total_seconds()),
                "online_ttl_sec": float(self.online_ttl.total_seconds()),
            }

    def reset_state(self, clear_db: bool = True) -> None:
        with self._lock:
            self._last_db_write_by_global.clear()
            self._last_seen_by_global.clear()
            self._first_seen_by_global.clear()
            self._seen_hits_by_global.clear()
            self._face_hits_by_global.clear()
            self._confirmed_globals.clear()
            self._last_center_y_norm_by_global.clear()
            self._last_crossing_by_global.clear()

        if not clear_db:
            return
        with db_conn(self.db_path) as conn:
            conn.execute("DELETE FROM sightings")
            conn.execute("DELETE FROM crossings")
            conn.execute("DELETE FROM face_photos")
            conn.commit()

    def get_online_ids(self) -> list[int]:
        now = datetime.now(timezone.utc)
        online_ids: list[int] = []
        with self._lock:
            for global_id, seen_at in self._last_seen_by_global.items():
                if (now - seen_at) <= self.online_ttl:
                    online_ids.append(global_id)
        return sorted(online_ids)

    def register_seen(self, global_id: int, now: datetime, face_confirmed: bool) -> None:
        should_write = False
        with self._lock:
            self._last_seen_by_global[global_id] = now
            self._seen_hits_by_global[global_id] = self._seen_hits_by_global.get(global_id, 0) + 1
            if face_confirmed:
                self._face_hits_by_global[global_id] = self._face_hits_by_global.get(global_id, 0) + 1
            first_seen = self._first_seen_by_global.get(global_id)
            if first_seen is None:
                first_seen = now
                self._first_seen_by_global[global_id] = now

            is_confirmed = global_id in self._confirmed_globals
            if not is_confirmed:
                enough_hits = self._seen_hits_by_global[global_id] >= self.count_confirm_min_hits
                enough_age = (now - first_seen) >= self.count_confirm_min_age
                face_hits = self._face_hits_by_global.get(global_id, 0)
                enough_face = (not self.unique_require_face_for_count) or (
                    face_hits >= self.face_confirm_min_hits
                )
                # Fallback for side-profile/brief passers: long-enough stable track can confirm
                # even without frontal face, when face requirement is enabled.
                no_face_fallback = (
                    self.count_confirm_no_face_fallback_enabled
                    and
                    self.unique_require_face_for_count
                    and (face_hits < self.face_confirm_min_hits)
                    and ((now - first_seen) >= self.count_confirm_no_face_age)
                )
                if enough_hits and enough_age and enough_face:
                    self._confirmed_globals.add(global_id)
                elif enough_hits and enough_age and no_face_fallback:
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

    def merge_global_ids(self, from_global_id: int, to_global_id: int) -> None:
        if from_global_id == to_global_id:
            return
        with self._lock:
            self._last_db_write_by_global[to_global_id] = max(
                self._last_db_write_by_global.get(to_global_id, datetime.min.replace(tzinfo=timezone.utc)),
                self._last_db_write_by_global.get(from_global_id, datetime.min.replace(tzinfo=timezone.utc)),
            )
            self._last_seen_by_global[to_global_id] = max(
                self._last_seen_by_global.get(to_global_id, datetime.min.replace(tzinfo=timezone.utc)),
                self._last_seen_by_global.get(from_global_id, datetime.min.replace(tzinfo=timezone.utc)),
            )
            first_to = self._first_seen_by_global.get(to_global_id)
            first_from = self._first_seen_by_global.get(from_global_id)
            if first_to is None:
                if first_from is not None:
                    self._first_seen_by_global[to_global_id] = first_from
            elif first_from is not None:
                self._first_seen_by_global[to_global_id] = min(first_to, first_from)
            self._seen_hits_by_global[to_global_id] = (
                self._seen_hits_by_global.get(to_global_id, 0)
                + self._seen_hits_by_global.get(from_global_id, 0)
            )
            self._face_hits_by_global[to_global_id] = (
                self._face_hits_by_global.get(to_global_id, 0)
                + self._face_hits_by_global.get(from_global_id, 0)
            )
            if from_global_id in self._confirmed_globals:
                self._confirmed_globals.add(to_global_id)
            if from_global_id in self._last_center_y_norm_by_global:
                self._last_center_y_norm_by_global[to_global_id] = self._last_center_y_norm_by_global[
                    from_global_id
                ]
            if from_global_id in self._last_crossing_by_global:
                self._last_crossing_by_global[to_global_id] = self._last_crossing_by_global[
                    from_global_id
                ]

            self._last_db_write_by_global.pop(from_global_id, None)
            self._last_seen_by_global.pop(from_global_id, None)
            self._first_seen_by_global.pop(from_global_id, None)
            self._seen_hits_by_global.pop(from_global_id, None)
            self._face_hits_by_global.pop(from_global_id, None)
            self._confirmed_globals.discard(from_global_id)
            self._last_center_y_norm_by_global.pop(from_global_id, None)
            self._last_crossing_by_global.pop(from_global_id, None)

        with db_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE sightings SET global_id = ? WHERE global_id = ?",
                (to_global_id, from_global_id),
            )
            conn.execute(
                "UPDATE crossings SET global_id = ? WHERE global_id = ?",
                (to_global_id, from_global_id),
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

        online_ids = self.get_online_ids()

        return {
            "timestamp_utc": now.isoformat(),
            "line_crossing_enabled": self.enable_line_crossing,
            "line_y_ratio": self.line_y_ratio,
            "count_confirm_min_age_sec": self.get_confirm_min_age_sec(),
            "count_confirm_no_face_fallback_enabled": self.get_no_face_fallback_enabled(),
            "unique_require_face_for_count": self.unique_require_face_for_count,
            "face_confirm_min_hits": self.face_confirm_min_hits,
            "online_count": len(online_ids),
            "online_global_ids": sorted(online_ids),
            "unique_last_hour": int(hour_row[0] or 0),
            "unique_last_24h": int(day_row[0] or 0),
            "entries_last_hour": int(in_hour_row[0] or 0),
            "exits_last_hour": int(out_hour_row[0] or 0),
            "entries_last_24h": int(in_day_row[0] or 0),
            "exits_last_24h": int(out_day_row[0] or 0),
        }
