from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np


@dataclass
class IdentityState:
    embedding: np.ndarray
    last_seen: datetime


class ReIDService:
    def __init__(
        self,
        match_threshold: float,
        max_absence_sec: float,
        track_ttl_sec: float,
    ) -> None:
        self.match_threshold = match_threshold
        self.max_absence = timedelta(seconds=max_absence_sec)
        self.track_ttl = timedelta(seconds=track_ttl_sec)
        self._next_global_id = 1
        self._track_to_global: dict[int, int] = {}
        self._track_last_seen: dict[int, datetime] = {}
        self._identities: dict[int, IdentityState] = {}

    def assign_global_id(
        self,
        track_id: int,
        bbox_xyxy: tuple[int, int, int, int],
        frame_bgr: np.ndarray,
        now: datetime,
    ) -> int:
        self._cleanup_tracks(now)
        if track_id in self._track_to_global:
            global_id = self._track_to_global[track_id]
            self._track_last_seen[track_id] = now
            embedding = self._extract_embedding(frame_bgr, bbox_xyxy)
            self._update_identity(global_id, embedding, now)
            return global_id

        embedding = self._extract_embedding(frame_bgr, bbox_xyxy)
        global_id = self._match_identity(embedding, now)
        if global_id is None:
            global_id = self._next_global_id
            self._next_global_id += 1
            self._identities[global_id] = IdentityState(embedding=embedding, last_seen=now)
        else:
            self._update_identity(global_id, embedding, now)

        self._track_to_global[track_id] = global_id
        self._track_last_seen[track_id] = now
        return global_id

    def _cleanup_tracks(self, now: datetime) -> None:
        stale = [
            track_id
            for track_id, last_seen in self._track_last_seen.items()
            if (now - last_seen) > self.track_ttl
        ]
        for track_id in stale:
            self._track_last_seen.pop(track_id, None)
            self._track_to_global.pop(track_id, None)

    def _match_identity(self, embedding: np.ndarray, now: datetime) -> int | None:
        best_id: int | None = None
        best_score = -1.0
        for global_id, state in self._identities.items():
            if (now - state.last_seen) > self.max_absence:
                continue
            score = self._cosine_similarity(embedding, state.embedding)
            if score > best_score:
                best_score = score
                best_id = global_id
        if best_score >= self.match_threshold:
            return best_id
        return None

    def _update_identity(self, global_id: int, embedding: np.ndarray, now: datetime) -> None:
        state = self._identities.get(global_id)
        if state is None:
            self._identities[global_id] = IdentityState(embedding=embedding, last_seen=now)
            return
        # Smooth appearance shifts while keeping identity stable.
        state.embedding = 0.8 * state.embedding + 0.2 * embedding
        state.embedding = self._normalize(state.embedding)
        state.last_seen = now

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
