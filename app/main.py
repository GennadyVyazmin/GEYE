import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .analytics import AnalyticsService
from .config import settings
from .db import db_conn, init_db
from .photo_gallery import PhotoGalleryService
from .reid import ReIDService
from .video_processor import VideoProcessor


app = FastAPI(title="GEYE Video Analytics")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


def load_next_global_id(db_path: str) -> int:
    with db_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT COALESCE(MAX(global_id), 0) FROM sightings
            UNION ALL
            SELECT COALESCE(MAX(global_id), 0) FROM crossings
            UNION ALL
            SELECT COALESCE(MAX(global_id), 0) FROM face_photos
            UNION ALL
            SELECT COALESCE(MAX(global_id), 0) FROM person_registry
            """
        ).fetchall()
    max_id = 0
    for row in rows:
        max_id = max(max_id, int(row[0] or 0))
    return max_id + 1


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
    unique_require_face_for_count=settings.unique_require_face_for_count,
    face_confirm_min_hits=settings.face_confirm_min_hits,
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
    face_embedder=settings.face_embedder,
    face_embedder_ctx_id=settings.face_embedder_ctx_id,
    face_lock_match_threshold=settings.face_lock_match_threshold,
    face_lock_margin=settings.face_lock_margin,
    face_min_score=settings.face_min_score,
    face_rebind_match_threshold=settings.face_rebind_match_threshold,
    face_rebind_margin=settings.face_rebind_margin,
    face_profiles_refresh_sec=settings.face_profiles_refresh_sec,
    face_profile_bank_per_id=settings.face_profile_bank_per_id,
    face_rebind_min_votes=settings.face_rebind_min_votes,
    face_rebind_cluster_delta=settings.face_rebind_cluster_delta,
    face_prefer_locked_delta=settings.face_prefer_locked_delta,
    face_global_dedup_threshold=settings.face_global_dedup_threshold,
    face_global_dedup_unknown_threshold=settings.face_global_dedup_unknown_threshold,
    face_reentry_merge_threshold=settings.face_reentry_merge_threshold,
    face_global_dedup_interval_sec=settings.face_global_dedup_interval_sec,
    face_stable_sim_threshold=settings.face_stable_sim_threshold,
    face_stable_min_hits=settings.face_stable_min_hits,
    face_locked_relaxed_threshold=settings.face_locked_relaxed_threshold,
)
reid = ReIDService(
    match_threshold=settings.reid_match_threshold,
    weak_match_threshold=settings.reid_weak_match_threshold,
    weak_margin=settings.reid_weak_margin,
    weak_match_recency_sec=settings.reid_weak_match_recency_sec,
    max_absence_sec=settings.reid_max_absence_sec,
    track_ttl_sec=settings.track_ttl_sec,
    start_global_id=load_next_global_id(settings.db_path),
)
processor = VideoProcessor(
    rtsp_url=settings.rtsp_url,
    model_path=settings.model_path,
    tracker_config_path=settings.tracker_config_path,
    conf=settings.detector_conf,
    iou=settings.detector_iou,
    min_person_box_height_px=settings.min_person_box_height_px,
    min_person_box_area_ratio=settings.min_person_box_area_ratio,
    frame_max_width=settings.frame_max_width,
    process_every_n_frames=settings.process_every_n_frames,
    jpeg_quality=settings.jpeg_quality,
    rtsp_low_latency_mode=settings.rtsp_low_latency_mode,
    rtsp_drain_grabs=settings.rtsp_drain_grabs,
    analytics=analytics,
    gallery=gallery,
    reid=reid,
)


class ReIDThresholdPayload(BaseModel):
    match_threshold: float


class ConfirmAgePayload(BaseModel):
    min_age_sec: float


class RegisterPersonPayload(BaseModel):
    global_id: int
    display_name: str
    note: str = ""


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
    reid.reset_state(reset_counter=False)
    return {"status": "ok", "message": "counters, tracks, and DB events reset"}


@app.post("/api/reset-all")
async def reset_all():
    analytics.reset_state(clear_db=True)
    gallery.reset_state(clear_files=True)
    with db_conn(settings.db_path) as conn:
        conn.execute("DELETE FROM person_registry")
        conn.execute("DELETE FROM person_face_profiles")
        conn.commit()
    reid.reset_state(reset_counter=True, start_global_id=1)
    return {
        "status": "ok",
        "message": "full reset completed (events, registry, tracks, global_id counter)",
    }


@app.get("/api/reid")
async def get_reid():
    return {"match_threshold": reid.get_match_threshold()}


@app.post("/api/reid")
async def set_reid(payload: ReIDThresholdPayload):
    if payload.match_threshold < 0.0 or payload.match_threshold > 1.0:
        raise HTTPException(status_code=400, detail="match_threshold must be in [0.0, 1.0]")
    value = reid.set_match_threshold(payload.match_threshold)
    return {"status": "ok", "match_threshold": value}


@app.get("/api/confirm-age")
async def get_confirm_age():
    return {"min_age_sec": analytics.get_confirm_min_age_sec()}


@app.post("/api/confirm-age")
async def set_confirm_age(payload: ConfirmAgePayload):
    if payload.min_age_sec < 0.0 or payload.min_age_sec > 20.0:
        raise HTTPException(status_code=400, detail="min_age_sec must be in [0.0, 20.0]")
    value = analytics.set_confirm_min_age_sec(payload.min_age_sec)
    return {"status": "ok", "min_age_sec": value}


@app.get("/api/gallery")
async def get_gallery(window: str = "online"):
    if window not in {"online", "hour", "day"}:
        raise HTTPException(status_code=400, detail="window must be online|hour|day")
    online_ids = analytics.get_online_ids()
    return JSONResponse(gallery.list_people(window=window, online_ids=online_ids))


@app.get("/api/people/{global_id}")
async def get_person(global_id: int):
    with db_conn(settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT global_id, display_name, note, created_ts, updated_ts
            FROM person_registry
            WHERE global_id = ?
            """,
            (global_id,),
        ).fetchone()
    if row is None:
        return {"registered": False, "global_id": global_id}
    return {
        "registered": True,
        "global_id": int(row[0]),
        "display_name": str(row[1]),
        "note": str(row[2]),
        "created_ts": str(row[3]),
        "updated_ts": str(row[4]),
    }


@app.post("/api/people/register")
async def register_person(payload: RegisterPersonPayload):
    name = payload.display_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="display_name is required")
    now = datetime.now(timezone.utc).isoformat()
    with db_conn(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO person_registry (global_id, display_name, note, created_ts, updated_ts)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(global_id) DO UPDATE SET
                display_name=excluded.display_name,
                note=excluded.note,
                updated_ts=excluded.updated_ts
            """,
            (payload.global_id, name, payload.note.strip(), now, now),
        )
        conn.commit()
    profile_ok = gallery.upsert_face_profile_for_global(payload.global_id)
    return {
        "status": "ok",
        "global_id": payload.global_id,
        "display_name": name,
        "face_profile_ready": profile_ok,
    }


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
