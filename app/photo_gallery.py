from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .db import db_conn


class PhotoGalleryService:
    def __init__(
        self,
        db_path: str,
        session_id: str,
        capture_dir: Path,
        photo_update_interval_sec: float,
        gallery_limit: int,
    ) -> None:
        self.db_path = db_path
        self.session_id = session_id
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.photo_update_interval = timedelta(seconds=max(0.5, photo_update_interval_sec))
        self.gallery_limit = max(1, gallery_limit)
        self._lock = threading.Lock()
        self._last_saved_by_global: dict[int, datetime] = {}
        self._best_score_by_global: dict[int, float] = {}

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)

    def reset_state(self, clear_files: bool = False) -> None:
        with self._lock:
            self._last_saved_by_global.clear()
            self._best_score_by_global.clear()
        if clear_files:
            for p in self.capture_dir.glob("*.jpg"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def register_detection(
        self,
        global_id: int,
        frame_bgr: np.ndarray,
        person_bbox_xyxy: tuple[int, int, int, int],
        now: datetime,
    ) -> None:
        person_crop = self._crop(frame_bgr, person_bbox_xyxy)
        if person_crop is None:
            return

        face_crop, score = self._extract_best_face(person_crop)
        if face_crop is None:
            return

        with self._lock:
            last_saved = self._last_saved_by_global.get(global_id)
            best_score = self._best_score_by_global.get(global_id, 0.0)
            if last_saved is not None and (now - last_saved) < self.photo_update_interval:
                return
            # Save only when quality is at least comparable to previous best.
            if score + 0.02 < best_score:
                return
            self._last_saved_by_global[global_id] = now
            self._best_score_by_global[global_id] = max(best_score, score)

        filename = f"g{global_id}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        abs_path = self.capture_dir / filename
        ok = cv2.imwrite(str(abs_path), face_crop)
        if not ok:
            return

        with db_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO face_photos (ts, session_id, global_id, image_name, face_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now.isoformat(), self.session_id, global_id, filename, float(score)),
            )
            conn.commit()

    def list_people(self, window: str, online_ids: list[int]) -> dict:
        now = datetime.now(timezone.utc)
        cutoff: datetime | None
        if window == "online":
            cutoff = None
            ids = sorted(set(online_ids))
        elif window == "hour":
            cutoff = now - timedelta(hours=1)
            ids = self._load_unique_ids_since(cutoff)
        elif window == "day":
            cutoff = now - timedelta(hours=24)
            ids = self._load_unique_ids_since(cutoff)
        else:
            raise ValueError("window must be one of: online, hour, day")

        ids = ids[: self.gallery_limit]
        items = []
        for global_id in ids:
            photo = self._pick_photo(global_id, cutoff)
            item = {
                "global_id": global_id,
                "image_url": None,
                "photo_ts": None,
                "has_frontal_face": False,
            }
            if photo is not None:
                item["image_url"] = f"/captures/{photo['image_name']}"
                item["photo_ts"] = photo["ts"]
                item["has_frontal_face"] = True
            items.append(item)
        return {"window": window, "count": len(items), "items": items}

    def _load_unique_ids_since(self, cutoff: datetime) -> list[int]:
        with db_conn(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT global_id
                FROM sightings
                WHERE ts >= ?
                ORDER BY global_id ASC
                LIMIT ?
                """,
                (cutoff.isoformat(), self.gallery_limit),
            ).fetchall()
        return [int(r[0]) for r in rows]

    def _pick_photo(self, global_id: int, cutoff: datetime | None) -> dict | None:
        with db_conn(self.db_path) as conn:
            if cutoff is None:
                row = conn.execute(
                    """
                    SELECT ts, image_name
                    FROM face_photos
                    WHERE global_id = ?
                    ORDER BY face_score DESC, ts DESC
                    LIMIT 1
                    """,
                    (global_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT ts, image_name
                    FROM face_photos
                    WHERE global_id = ? AND ts >= ?
                    ORDER BY face_score DESC, ts DESC
                    LIMIT 1
                    """,
                    (global_id, cutoff.isoformat()),
                ).fetchone()
                if row is None:
                    row = conn.execute(
                        """
                        SELECT ts, image_name
                        FROM face_photos
                        WHERE global_id = ?
                        ORDER BY face_score DESC, ts DESC
                        LIMIT 1
                        """,
                        (global_id,),
                    ).fetchone()
        if row is None:
            return None
        return {"ts": str(row[0]), "image_name": str(row[1])}

    @staticmethod
    def _crop(frame_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _extract_best_face(self, person_crop: np.ndarray) -> tuple[np.ndarray | None, float]:
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(faces) == 0:
            return None, 0.0

        h, w = person_crop.shape[:2]
        best = None
        best_score = -1.0
        cx_ref = w / 2.0
        cy_ref = h / 3.0
        for x, y, fw, fh in faces:
            area_ratio = (fw * fh) / float(max(1, w * h))
            fx = x + (fw / 2.0)
            fy = y + (fh / 2.0)
            dx = abs(fx - cx_ref) / max(1.0, w)
            dy = abs(fy - cy_ref) / max(1.0, h)
            center_score = max(0.0, 1.0 - (dx + dy))
            score = (0.65 * area_ratio) + (0.35 * center_score)
            if score > best_score:
                best_score = score
                best = (x, y, fw, fh)
        if best is None:
            return None, 0.0

        x, y, fw, fh = best
        margin_x = int(0.2 * fw)
        margin_y = int(0.25 * fh)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + fw + margin_x)
        y2 = min(h, y + fh + margin_y)
        face_crop = person_crop[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, 0.0
        return face_crop, float(best_score)
