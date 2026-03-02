from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    server_ip: str = "127.0.0.1"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    rtsp_url: str = "rtsp://user:password@camera-ip:554/stream1"
    model_path: str = "yolov8n.pt"
    detector_conf: float = 0.35
    detector_iou: float = 0.5
    frame_max_width: int = 960
    process_every_n_frames: int = 2
    jpeg_quality: int = 75
    min_db_event_interval_sec: float = 2.0
    online_ttl_sec: float = 2.0
    enable_line_crossing: bool = False
    line_y_ratio: float = 0.55
    crossing_debounce_sec: float = 2.0
    reid_match_threshold: float = 0.85
    reid_max_absence_sec: float = 300.0
    track_ttl_sec: float = 3.0
    db_path: str = "analytics.db"


settings = Settings()
