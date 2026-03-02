import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analytics import AnalyticsService
from .config import settings
from .db import init_db
from .reid import ReIDService
from .video_processor import VideoProcessor


app = FastAPI(title="GEYE Video Analytics")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

session_id = str(uuid.uuid4())
init_db(settings.db_path)
analytics = AnalyticsService(
    db_path=settings.db_path,
    session_id=session_id,
    min_db_event_interval_sec=settings.min_db_event_interval_sec,
    online_ttl_sec=settings.online_ttl_sec,
    enable_line_crossing=settings.enable_line_crossing,
    line_y_ratio=settings.line_y_ratio,
    crossing_debounce_sec=settings.crossing_debounce_sec,
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
    analytics=analytics,
    reid=reid,
)


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
