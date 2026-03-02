import threading
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .analytics import AnalyticsService
from .reid import ReIDService


class VideoProcessor:
    def __init__(
        self,
        rtsp_url: str,
        model_path: str,
        conf: float,
        iou: float,
        analytics: AnalyticsService,
        reid: ReIDService,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.conf = conf
        self.iou = iou
        self.analytics = analytics
        self.reid = reid
        self.model = YOLO(model_path)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_jpeg: Optional[bytes] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._last_jpeg

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(2)
                continue

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                results = self.model.track(
                    source=frame,
                    persist=True,
                    classes=[0],  # person
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                )
                result = results[0]
                rendered = frame.copy()
                frame_h, frame_w = rendered.shape[:2]
                line_y = int(frame_h * self.analytics.line_y_ratio)
                cv2.line(rendered, (0, line_y), (frame_w, line_y), (0, 180, 255), 2)

                now = datetime.now(timezone.utc)
                if result.boxes is not None and result.boxes.id is not None:
                    ids = [int(x) for x in result.boxes.id.cpu().tolist()]
                    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
                    for idx, track_id in enumerate(ids):
                        x1, y1, x2, y2 = [int(v) for v in boxes[idx].tolist()]
                        global_id = self.reid.assign_global_id(
                            track_id=track_id,
                            bbox_xyxy=(x1, y1, x2, y2),
                            frame_bgr=frame,
                            now=now,
                        )
                        self.analytics.register_seen(global_id, now)
                        center_y = (y1 + y2) / 2.0
                        self.analytics.register_position(
                            global_id=global_id,
                            center_y_px=center_y,
                            frame_height_px=frame_h,
                            now=now,
                        )

                        cv2.rectangle(rendered, (x1, y1), (x2, y2), (80, 220, 120), 2)
                        label = f"G{global_id} T{track_id}"
                        cv2.putText(
                            rendered,
                            label,
                            (x1, max(18, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (80, 220, 120),
                            2,
                            cv2.LINE_AA,
                        )

                ok_jpg, jpg = cv2.imencode(".jpg", rendered)
                if ok_jpg:
                    with self._frame_lock:
                        self._last_jpeg = jpg.tobytes()

            cap.release()
            time.sleep(1)
