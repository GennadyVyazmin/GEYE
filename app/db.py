import sqlite3
from contextlib import contextmanager


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT NOT NULL,
                global_id INTEGER NOT NULL
            )
            """
        )
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sightings)").fetchall()
        }
        if "global_id" not in columns:
            # Backward compatibility with earlier MVP schema.
            conn.execute("ALTER TABLE sightings ADD COLUMN global_id INTEGER")
            conn.execute(
                "UPDATE sightings SET global_id = COALESCE(global_id, track_id, -1)"
            )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sightings_ts
            ON sightings(ts)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sightings_session_global
            ON sightings(session_id, global_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crossings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT NOT NULL,
                global_id INTEGER NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('in', 'out'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crossings_ts
            ON crossings(ts)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_crossings_session_global
            ON crossings(session_id, global_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                session_id TEXT NOT NULL,
                global_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                face_score REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_face_photos_ts
            ON face_photos(ts)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_face_photos_global
            ON face_photos(global_id)
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def db_conn(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()
