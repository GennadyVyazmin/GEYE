import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .analytics import AnalyticsService
from .config import settings
from .db import init_db
from .photo_gallery import PhotoGalleryService
from .reid import ReIDService
from .video_processor import VideoProcessor


app = FastAPI(title="GEYE Video Analytics")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

session_id = str(uuid.uuid4())
init_db(settings.db_path)
capture_dir = Path(settings.photo_dir)
if not capture_dir.is_absolute():
    capture_dir = (Path.cwd() / capture_dir).resolve()
capture_dir.mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=str(capture_dir)), name="captures")

analytics = AnalyticsService(
    db_path=settings.db_path,
    session_id=session_id,
    min_db_event_interval_sec=settings.min_db_event_interval_sec,
    count_confirm_min_hits=settings.count_confirm_min_hits,
    count_confirm_min_age_sec=settings.count_confirm_min_age_sec,
    online_ttl_sec=settings.online_ttl_sec,
    enable_line_crossing=settings.enable_line_crossing,
    line_y_ratio=settings.line_y_ratio,
    crossing_debounce_sec=settings.crossing_debounce_sec,
)
gallery = PhotoGalleryService(
    db_path=settings.db_path,
    session_id=session_id,
    capture_dir=capture_dir,
    photo_capture_once_per_id=settings.photo_capture_once_per_id,
    photo_update_interval_sec=settings.photo_update_interval_sec,
    gallery_limit=settings.gallery_limit,
)
reid = ReIDService(
    match_threshold=settings.reid_match_threshold,
    max_absence_sec=settings.reid_max_absence_sec,
    track_ttl_sec=settings.track_ttl_sec,
)
processor = VideoProcessor(
    rtsp_url=settings.rtsp_url,
    model_path=settings.model_path,
    conf=settings.detector_conf,
    iou=settings.detector_iou,
    frame_max_width=settings.frame_max_width,
    process_every_n_frames=settings.process_every_n_frames,
    jpeg_quality=settings.jpeg_quality,
    analytics=analytics,
    gallery=gallery,
    reid=reid,
)


class ReIDThresholdPayload(BaseModel):
    match_threshold: float


@app.on_event("startup")
def on_startup() -> None:
    processor.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    processor.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/stats")
async def stats():
    return JSONResponse(analytics.get_stats())


@app.post("/api/reset")
async def reset():
    analytics.reset_state(clear_db=True)
    gallery.reset_state(clear_files=True)
    reid.reset_state()
    return {"status": "ok", "message": "counters, tracks, and DB events reset"}


@app.get("/api/reid")
async def get_reid():
    return {"match_threshold": reid.get_match_threshold()}


@app.post("/api/reid")
async def set_reid(payload: ReIDThresholdPayload):
    if payload.match_threshold < 0.0 or payload.match_threshold > 1.0:
        raise HTTPException(status_code=400, detail="match_threshold must be in [0.0, 1.0]")
    value = reid.set_match_threshold(payload.match_threshold)
    return {"status": "ok", "match_threshold": value}


@app.get("/api/gallery")
async def get_gallery(window: str = "online"):
    if window not in {"online", "hour", "day"}:
        raise HTTPException(status_code=400, detail="window must be online|hour|day")
    online_ids = analytics.get_online_ids()
    return JSONResponse(gallery.list_people(window=window, online_ids=online_ids))


@app.get("/health")
async def health():
    return {"status": "ok", "session_id": session_id}


@app.get("/video_feed")
async def video_feed():
    async def stream():
        while True:
            frame = processor.get_latest_jpeg()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/frame.jpg")
async def frame_jpg():
    frame = processor.get_latest_jpeg()
    if frame is None:
        return Response(status_code=204)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )
