import threading
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from .analytics import AnalyticsService
from .photo_gallery import PhotoGalleryService
from .reid import ReIDService


class VideoProcessor:
    def __init__(
        self,
        rtsp_url: str,
        model_path: str,
        tracker_config_path: str,
        conf: float,
        iou: float,
        min_person_box_height_px: int,
        min_person_box_area_ratio: float,
        frame_max_width: int,
        process_every_n_frames: int,
        jpeg_quality: int,
        rtsp_low_latency_mode: bool,
        rtsp_drain_grabs: int,
        analytics: AnalyticsService,
        gallery: PhotoGalleryService,
        reid: ReIDService,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.conf = conf
        self.iou = iou
        tpath = Path(tracker_config_path)
        self.tracker_config_path = str(tpath.resolve()) if tpath.exists() else None
        self.min_person_box_height_px = max(1, int(min_person_box_height_px))
        self.min_person_box_area_ratio = max(0.0, float(min_person_box_area_ratio))
        self.frame_max_width = max(320, frame_max_width)
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.jpeg_quality = max(40, min(95, jpeg_quality))
        self.rtsp_low_latency_mode = rtsp_low_latency_mode
        self.rtsp_drain_grabs = max(0, rtsp_drain_grabs)
        self.analytics = analytics
        self.gallery = gallery
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
        if self.rtsp_low_latency_mode:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
            )
        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                time.sleep(2)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame_counter = 0

            while not self._stop_event.is_set():
                ok, frame = self._read_latest_frame(cap)
                if not ok:
                    break
                frame_counter += 1
                if (frame_counter % self.process_every_n_frames) != 0:
                    continue

                frame = self._resize_max_width(frame, self.frame_max_width)

                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config_path,
                    classes=[0],  # person
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                )
                result = results[0]
                rendered = frame.copy()
                frame_h, frame_w = rendered.shape[:2]
                if self.analytics.enable_line_crossing:
                    line_y = int(frame_h * self.analytics.line_y_ratio)
                    cv2.line(rendered, (0, line_y), (frame_w, line_y), (0, 180, 255), 2)

                now = datetime.now(timezone.utc)
                if result.boxes is not None and result.boxes.id is not None:
                    ids = [int(x) for x in result.boxes.id.cpu().tolist()]
                    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
                    for idx, track_id in enumerate(ids):
                        x1, y1, x2, y2 = [int(v) for v in boxes[idx].tolist()]
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)
                        area_ratio = (bw * bh) / float(max(1, frame_w * frame_h))
                        if bh < self.min_person_box_height_px:
                            continue
                        if area_ratio < self.min_person_box_area_ratio:
                            continue
                        global_id = self.reid.assign_global_id(
                            track_id=track_id,
                            bbox_xyxy=(x1, y1, x2, y2),
                            frame_bgr=frame,
                            now=now,
                        )
                        face_confirmed = self.gallery.register_detection(
                            global_id=global_id,
                            frame_bgr=frame,
                            person_bbox_xyxy=(x1, y1, x2, y2),
                            now=now,
                        )
                        self.analytics.register_seen(global_id, now, face_confirmed=face_confirmed)
                        if self.analytics.enable_line_crossing:
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

                ok_jpg, jpg = cv2.imencode(
                    ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                if ok_jpg:
                    with self._frame_lock:
                        self._last_jpeg = jpg.tobytes()

            cap.release()
            time.sleep(1)

    @staticmethod
    def _resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / float(w)
        target_h = max(1, int(h * scale))
        return cv2.resize(frame, (max_width, target_h), interpolation=cv2.INTER_AREA)

    def _read_latest_frame(self, cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
        # Grab several frames and decode only the latest to avoid RTSP lag buildup.
        ok = cap.grab()
        if not ok:
            return False, None
        for _ in range(self.rtsp_drain_grabs):
            if not cap.grab():
                break
        ok, frame = cap.retrieve()
        if not ok:
            return False, None
        return True, frame
