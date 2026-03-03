from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .db import db_conn


@dataclass
class FaceDetectionResult:
    face_confirmed: bool
    locked_global_id: int | None


class PhotoGalleryService:
    def __init__(
        self,
        db_path: str,
        session_id: str,
        capture_dir: Path,
        photo_capture_once_per_id: bool,
        photo_update_interval_sec: float,
        gallery_limit: int,
        face_lock_match_threshold: float,
        face_lock_margin: float,
        face_min_score: float,
        face_rebind_match_threshold: float,
        face_rebind_margin: float,
        face_profiles_refresh_sec: float,
        face_profile_bank_per_id: int,
        face_rebind_min_votes: int,
        face_rebind_cluster_delta: float,
    ) -> None:
        self.db_path = db_path
        self.session_id = session_id
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.photo_capture_once_per_id = photo_capture_once_per_id
        self.photo_update_interval = timedelta(seconds=max(0.5, photo_update_interval_sec))
        self.gallery_limit = max(1, gallery_limit)
        self.face_lock_match_threshold = max(0.0, min(1.0, face_lock_match_threshold))
        self.face_lock_margin = max(0.0, min(0.5, face_lock_margin))
        self.face_min_score = max(0.0, min(1.0, face_min_score))
        self.face_rebind_match_threshold = max(0.0, min(1.0, face_rebind_match_threshold))
        self.face_rebind_margin = max(0.0, min(0.5, face_rebind_margin))
        self.face_profiles_refresh_interval = timedelta(seconds=max(2.0, face_profiles_refresh_sec))
        self.face_profile_bank_per_id = max(1, int(face_profile_bank_per_id))
        self.face_rebind_min_votes = max(1, int(face_rebind_min_votes))
        self.face_rebind_cluster_delta = max(0.0, min(0.2, float(face_rebind_cluster_delta)))
        self._lock = threading.Lock()
        self._last_saved_by_global: dict[int, datetime] = {}
        self._best_score_by_global: dict[int, float] = {}
        self._has_photo_by_global: set[int] = set()
        self._locked_profiles: dict[int, np.ndarray] = {}
        self._photo_profiles: dict[int, list[np.ndarray]] = {}
        self._last_photo_profiles_reload: datetime = datetime.min.replace(tzinfo=timezone.utc)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        eyes_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self.eye_detector = cv2.CascadeClassifier(eyes_path)
        self._reload_locked_profiles()
        self._reload_photo_profiles()

    def reset_state(self, clear_files: bool = False) -> None:
        with self._lock:
            self._last_saved_by_global.clear()
            self._best_score_by_global.clear()
            self._has_photo_by_global.clear()
            self._locked_profiles.clear()
            self._photo_profiles.clear()
            self._last_photo_profiles_reload = datetime.min.replace(tzinfo=timezone.utc)
        if clear_files:
            for p in self.capture_dir.glob("*.jpg"):
                try:
                    p.unlink()
                except OSError:
                    pass
        self._reload_locked_profiles()

    def register_detection(
        self,
        global_id: int,
        frame_bgr: np.ndarray,
        person_bbox_xyxy: tuple[int, int, int, int],
        now: datetime,
    ) -> FaceDetectionResult:
        person_crop = self._crop(frame_bgr, person_bbox_xyxy)
        if person_crop is None:
            return FaceDetectionResult(face_confirmed=False, locked_global_id=None)

        face_crop, score = self._extract_best_face(person_crop)
        if face_crop is None or score < self.face_min_score:
            return FaceDetectionResult(face_confirmed=False, locked_global_id=None)

        face_embedding = self._extract_face_embedding(face_crop)
        locked_global_id = self._match_locked_profile(face_embedding)
        if locked_global_id is None:
            self._maybe_reload_photo_profiles(now)
            locked_global_id = self._match_photo_profile(face_embedding, current_global_id=global_id)

        with self._lock:
            if self.photo_capture_once_per_id and global_id in self._has_photo_by_global:
                return FaceDetectionResult(face_confirmed=True, locked_global_id=locked_global_id)
            last_saved = self._last_saved_by_global.get(global_id)
            best_score = self._best_score_by_global.get(global_id, 0.0)
            if last_saved is not None and (now - last_saved) < self.photo_update_interval:
                return FaceDetectionResult(face_confirmed=True, locked_global_id=locked_global_id)
            # Save only when quality is at least comparable to previous best.
            if score + 0.02 < best_score:
                return FaceDetectionResult(face_confirmed=True, locked_global_id=locked_global_id)
            self._last_saved_by_global[global_id] = now
            self._best_score_by_global[global_id] = max(best_score, score)

        filename = f"g{global_id}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        abs_path = self.capture_dir / filename
        ok = cv2.imwrite(str(abs_path), face_crop)
        if not ok:
            return FaceDetectionResult(face_confirmed=True, locked_global_id=locked_global_id)

        with db_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO face_photos (ts, session_id, global_id, image_name, face_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now.isoformat(), self.session_id, global_id, filename, float(score)),
            )
            conn.commit()
        with self._lock:
            self._has_photo_by_global.add(global_id)
        if locked_global_id is None:
            self._maybe_reload_photo_profiles(now, force=True)
        return FaceDetectionResult(face_confirmed=True, locked_global_id=locked_global_id)

    def upsert_face_profile_for_global(self, global_id: int) -> bool:
        with db_conn(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT image_name
                FROM face_photos
                WHERE global_id = ?
                ORDER BY face_score DESC, ts DESC
                LIMIT 1
                """,
                (global_id,),
            ).fetchone()
        if row is None:
            return False
        image_name = str(row[0])
        img_path = self.capture_dir / image_name
        if not img_path.exists():
            return False
        image = cv2.imread(str(img_path))
        if image is None:
            return False
        emb = self._extract_face_embedding(image)
        now = datetime.now(timezone.utc).isoformat()
        emb_json = json.dumps(emb.tolist())
        with db_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO person_face_profiles (global_id, embedding_json, updated_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(global_id) DO UPDATE SET
                    embedding_json=excluded.embedding_json,
                    updated_ts=excluded.updated_ts
                """,
                (global_id, emb_json, now),
            )
            conn.commit()
        self._reload_locked_profiles()
        self._reload_photo_profiles()
        return True

    def _reload_locked_profiles(self) -> None:
        with db_conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT global_id, embedding_json FROM person_face_profiles"
            ).fetchall()
        profiles: dict[int, np.ndarray] = {}
        for gid, emb_json in rows:
            try:
                arr = np.array(json.loads(str(emb_json)), dtype=np.float32)
                arr = self._normalize(arr)
                if arr.size > 0:
                    profiles[int(gid)] = arr
            except Exception:
                continue
        with self._lock:
            self._locked_profiles = profiles

    def _reload_photo_profiles(self) -> None:
        with db_conn(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT global_id, image_name
                FROM face_photos
                ORDER BY face_score DESC, ts DESC
                LIMIT 5000
                """
            ).fetchall()
        profiles: dict[int, list[np.ndarray]] = {}
        for global_id, image_name in rows:
            gid = int(global_id)
            current = profiles.get(gid)
            if current is None:
                current = []
                profiles[gid] = current
            if len(current) >= self.face_profile_bank_per_id:
                continue
            img_path = self.capture_dir / str(image_name)
            if not img_path.exists():
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            emb = self._extract_face_embedding(image)
            if emb.size == 0:
                continue
            current.append(emb)
            if len(profiles) >= self.gallery_limit * 3:
                break
        with self._lock:
            self._photo_profiles = profiles
            self._last_photo_profiles_reload = datetime.now(timezone.utc)

    def _maybe_reload_photo_profiles(self, now: datetime, force: bool = False) -> None:
        with self._lock:
            last_reload = self._last_photo_profiles_reload
        if not force and (now - last_reload) < self.face_profiles_refresh_interval:
            return
        self._reload_photo_profiles()

    def has_locked_profile(self, global_id: int) -> bool:
        with self._lock:
            return int(global_id) in self._locked_profiles

    def _match_locked_profile(self, face_embedding: np.ndarray) -> int | None:
        with self._lock:
            items = list(self._locked_profiles.items())
        if not items:
            return None
        best_gid = None
        best_score = -1.0
        second_score = -1.0
        for gid, emb in items:
            score = self._cosine_similarity(face_embedding, emb)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_gid = gid
            elif score > second_score:
                second_score = score
        margin = best_score - max(0.0, second_score)
        if best_score >= self.face_lock_match_threshold and margin >= self.face_lock_margin:
            return best_gid
        return None

    def _match_photo_profile(self, face_embedding: np.ndarray, current_global_id: int) -> int | None:
        with self._lock:
            items = list(self._photo_profiles.items())
        if not items:
            return None
        scored: list[tuple[int, float]] = []
        for gid, embs in items:
            if len(embs) == 0:
                continue
            sims = sorted((self._cosine_similarity(face_embedding, e) for e in embs), reverse=True)
            votes = min(self.face_rebind_min_votes, len(sims))
            score = float(np.mean(sims[:votes]))
            scored.append((int(gid), score))
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        best_gid, best_score = scored[0]
        if best_score < self.face_rebind_match_threshold:
            return None

        # If several IDs are almost equally close, collapse them to the oldest (min G-ID).
        # This removes duplicated IDs of the same person (e.g. G2/G3/G4 for one face).
        cluster = [
            gid
            for gid, score in scored
            if score >= self.face_rebind_match_threshold
            and (best_score - score) <= self.face_rebind_cluster_delta
        ]
        if cluster:
            return int(min(cluster))

        second_score = scored[1][1] if len(scored) > 1 else 0.0
        margin = best_score - max(0.0, second_score)
        if margin >= self.face_rebind_margin:
            return int(best_gid)
        return None

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
        registry_map = self._load_registry(ids)
        items = []
        for global_id in ids:
            photo = self._pick_photo(global_id, cutoff)
            reg = registry_map.get(global_id)
            item = {
                "global_id": global_id,
                "image_url": None,
                "photo_ts": None,
                "has_frontal_face": False,
                "registered": reg is not None,
                "display_name": (reg["display_name"] if reg else ""),
                "note": (reg["note"] if reg else ""),
            }
            if photo is not None:
                item["image_url"] = f"/captures/{photo['image_name']}"
                item["photo_ts"] = photo["ts"]
                item["has_frontal_face"] = True
            items.append(item)
        return {"window": window, "count": len(items), "items": items}

    def _load_registry(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        query = (
            f"SELECT global_id, display_name, note FROM person_registry "
            f"WHERE global_id IN ({placeholders})"
        )
        with db_conn(self.db_path) as conn:
            rows = conn.execute(query, tuple(ids)).fetchall()
        out: dict[int, dict] = {}
        for row in rows:
            out[int(row[0])] = {"display_name": str(row[1]), "note": str(row[2])}
        return out

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
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        with db_conn(self.db_path) as conn:
            if cutoff is None:
                recent_cutoff = minute_ago
            else:
                recent_cutoff = max(cutoff, minute_ago)
            row = conn.execute(
                """
                SELECT ts, image_name
                FROM face_photos
                WHERE global_id = ? AND ts >= ?
                ORDER BY face_score DESC, ts DESC
                LIMIT 1
                """,
                (global_id, recent_cutoff.isoformat()),
            ).fetchone()
            if row is not None:
                return {"ts": str(row[0]), "image_name": str(row[1])}
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
        faces_frontal = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(24, 24),
        )
        faces = list(faces_frontal)
        if len(faces) == 0:
            return None, 0.0

        h, w = person_crop.shape[:2]
        best = None
        best_score = -1.0
        cx_ref = w / 2.0
        cy_ref = h / 3.0
        for x, y, fw, fh in faces:
            area_ratio = (fw * fh) / float(max(1, w * h))
            aspect = fw / float(max(1, fh))
            if area_ratio < 0.02:
                continue
            if aspect < 0.72 or aspect > 1.42:
                continue
            # Face should be in upper portion of person crop; rejects many object false positives.
            if (y + (fh / 2.0)) > (h * 0.70):
                continue
            fx = x + (fw / 2.0)
            fy = y + (fh / 2.0)
            dx = abs(fx - cx_ref) / max(1.0, w)
            dy = abs(fy - cy_ref) / max(1.0, h)
            center_score = max(0.0, 1.0 - (dx + dy))
            face_roi = gray[y : y + fh, x : x + fw]
            eyes = self.eye_detector.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(10, 10),
            )
            if len(eyes) < 1 and area_ratio < 0.09:
                continue
            eye_score = min(1.0, len(eyes) / 2.0)
            score = (0.55 * area_ratio) + (0.25 * center_score) + (0.20 * eye_score)
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

    @staticmethod
    def _extract_face_embedding(face_crop_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 12], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        return PhotoGalleryService._normalize(hist)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.dot(a, b))
