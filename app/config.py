from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    server_ip: str = "127.0.0.1"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    rtsp_url: str = "rtsp://user:password@camera-ip:554/stream1"
    model_path: str = "yolov8n.pt"
    tracker_config_path: str = "app/tracker_bytetrack.yaml"
    detector_conf: float = 0.35
    detector_iou: float = 0.5
    min_person_box_height_px: int = 60
    min_person_box_area_ratio: float = 0.012
    frame_max_width: int = 960
    process_every_n_frames: int = 2
    jpeg_quality: int = 75
    rtsp_low_latency_mode: bool = True
    rtsp_drain_grabs: int = 3
    min_db_event_interval_sec: float = 2.0
    count_confirm_min_hits: int = 2
    count_confirm_min_age_sec: float = 1.5
    unique_require_face_for_count: bool = True
    face_confirm_min_hits: int = 1
    count_confirm_no_face_fallback_enabled: bool = True
    count_confirm_no_face_age_sec: float = 2.5
    online_ttl_sec: float = 2.0
    photo_dir: str = "captures"
    photo_capture_once_per_id: bool = True
    photo_update_interval_sec: float = 4.0
    gallery_limit: int = 120
    face_embedder: str = "auto"
    face_embedder_ctx_id: int = 0
    face_lock_match_threshold: float = 0.72
    face_lock_margin: float = 0.05
    face_min_score: float = 0.06
    face_rebind_match_threshold: float = 0.84
    face_rebind_margin: float = 0.04
    face_profiles_refresh_sec: float = 8.0
    face_profile_bank_per_id: int = 6
    face_rebind_min_votes: int = 2
    face_rebind_cluster_delta: float = 0.03
    face_prefer_locked_delta: float = 0.08
    face_global_dedup_threshold: float = 0.85
    face_global_dedup_unknown_threshold: float = 0.90
    face_reentry_merge_threshold: float = 0.80
    face_global_dedup_interval_sec: float = 3.0
    face_stable_sim_threshold: float = 0.80
    face_stable_min_hits: int = 2
    face_locked_relaxed_threshold: float = 0.78
    enable_line_crossing: bool = False
    line_y_ratio: float = 0.55
    crossing_debounce_sec: float = 2.0
    reid_match_threshold: float = 0.68
    reid_weak_match_threshold: float = 0.50
    reid_weak_margin: float = 0.08
    reid_weak_match_recency_sec: float = 120.0
    reid_max_absence_sec: float = 86400.0
    track_ttl_sec: float = 15.0
    db_path: str = "analytics.db"


settings = Settings()
