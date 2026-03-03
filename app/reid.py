from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

import cv2
import numpy as np


@dataclass
class IdentityState:
    embedding: np.ndarray
    last_seen: datetime
    center_norm_xy: tuple[float, float]


class ReIDService:
    def __init__(
        self,
        match_threshold: float,
        weak_match_threshold: float,
        weak_match_recency_sec: float,
        max_absence_sec: float,
        track_ttl_sec: float,
        start_global_id: int = 1,
    ) -> None:
        self.match_threshold = match_threshold
        self.weak_match_threshold = weak_match_threshold
        self.weak_match_recency = timedelta(seconds=weak_match_recency_sec)
        self.max_absence = timedelta(seconds=max_absence_sec)
        self.track_ttl = timedelta(seconds=track_ttl_sec)
        self._lock = threading.Lock()
        self._next_global_id = max(1, int(start_global_id))
        self._track_to_global: dict[int, int] = {}
        self._track_last_seen: dict[int, datetime] = {}
        self._identities: dict[int, IdentityState] = {}

    def get_match_threshold(self) -> float:
        with self._lock:
            return self.match_threshold

    def set_match_threshold(self, value: float) -> float:
        value = max(0.0, min(1.0, value))
        with self._lock:
            self.match_threshold = value
            return self.match_threshold

    def assign_global_id(
        self,
        track_id: int,
        bbox_xyxy: tuple[int, int, int, int],
        frame_bgr: np.ndarray,
        now: datetime,
    ) -> int:
        embedding = self._extract_embedding(frame_bgr, bbox_xyxy)
        center_norm_xy = self._bbox_center_norm(bbox_xyxy, frame_bgr.shape[1], frame_bgr.shape[0])
        with self._lock:
            self._cleanup_tracks(now)
            if track_id in self._track_to_global:
                global_id = self._track_to_global[track_id]
                self._track_last_seen[track_id] = now
                self._update_identity(global_id, embedding, now, center_norm_xy)
                return global_id

            global_id = self._match_identity(embedding, now, center_norm_xy)
            if global_id is None:
                global_id = self._next_global_id
                self._next_global_id += 1
                self._identities[global_id] = IdentityState(
                    embedding=embedding, last_seen=now, center_norm_xy=center_norm_xy
                )
            else:
                self._update_identity(global_id, embedding, now, center_norm_xy)

            self._track_to_global[track_id] = global_id
            self._track_last_seen[track_id] = now
            return global_id

    def reset_state(self, reset_counter: bool = False, start_global_id: int = 1) -> None:
        with self._lock:
            if reset_counter:
                self._next_global_id = max(1, int(start_global_id))
            self._track_to_global.clear()
            self._track_last_seen.clear()
            self._identities.clear()

    def _cleanup_tracks(self, now: datetime) -> None:
        stale = [
            track_id
            for track_id, last_seen in self._track_last_seen.items()
            if (now - last_seen) > self.track_ttl
        ]
        for track_id in stale:
            self._track_last_seen.pop(track_id, None)
            self._track_to_global.pop(track_id, None)

    def _match_identity(
        self, embedding: np.ndarray, now: datetime, center_norm_xy: tuple[float, float]
    ) -> int | None:
        best_id: int | None = None
        best_score = -1.0
        recent_candidates = 0
        near_best_id: int | None = None
        near_best_score = -1.0
        near_recent_candidates = 0
        for global_id, state in self._identities.items():
            if (now - state.last_seen) > self.max_absence:
                continue
            if (now - state.last_seen) <= self.weak_match_recency:
                recent_candidates += 1
            score = self._cosine_similarity(embedding, state.embedding)
            if score > best_score:
                best_score = score
                best_id = global_id
            dist = self._center_distance(center_norm_xy, state.center_norm_xy)
            if dist <= 0.22:
                if (now - state.last_seen) <= self.weak_match_recency:
                    near_recent_candidates += 1
                if score > near_best_score:
                    near_best_score = score
                    near_best_id = global_id
        if best_score >= self.match_threshold:
            return best_id
        # Soft fallback: if scene recently had essentially one candidate,
        # allow weaker appearance match to keep ID stable after pose/hood/back changes.
        if recent_candidates <= 1 and best_score >= self.weak_match_threshold:
            return best_id
        # Spatial fallback: in multi-person scenes prefer a nearby recent identity.
        if near_recent_candidates <= 1 and near_best_score >= self.weak_match_threshold:
            return near_best_id
        return None

    def _update_identity(
        self,
        global_id: int,
        embedding: np.ndarray,
        now: datetime,
        center_norm_xy: tuple[float, float],
    ) -> None:
        state = self._identities.get(global_id)
        if state is None:
            self._identities[global_id] = IdentityState(
                embedding=embedding,
                last_seen=now,
                center_norm_xy=center_norm_xy,
            )
            return
        # Smooth appearance shifts while keeping identity stable.
        state.embedding = 0.8 * state.embedding + 0.2 * embedding
        state.embedding = self._normalize(state.embedding)
        state.last_seen = now
        prev_x, prev_y = state.center_norm_xy
        x, y = center_norm_xy
        state.center_norm_xy = ((0.8 * prev_x) + (0.2 * x), (0.8 * prev_y) + (0.2 * y))

    @staticmethod
    def _extract_embedding(frame_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((512,), dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((512,), dtype=np.float32)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        return ReIDService._normalize(hist)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    @staticmethod
    def _bbox_center_norm(
        bbox_xyxy: tuple[int, int, int, int], frame_w: int, frame_h: int
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox_xyxy
        cx = ((x1 + x2) / 2.0) / float(max(1, frame_w))
        cy = ((y1 + y2) / 2.0) / float(max(1, frame_h))
        return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))

    @staticmethod
    def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float(np.sqrt((dx * dx) + (dy * dy)))
