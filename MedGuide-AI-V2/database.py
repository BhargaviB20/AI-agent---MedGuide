import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "medguide.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                age INTEGER,
                gender TEXT,
                location TEXT,
                symptoms TEXT,
                risk_level TEXT,
                final_response TEXT,
                response_time_seconds REAL
            )
        """)


def save_consultation(patient, risk_level, final_response, response_time):
    with _connect() as conn:
        conn.execute(
            """INSERT INTO consultations
               (timestamp, age, gender, location, symptoms, risk_level, final_response, response_time_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                patient["age"], patient["gender"], patient["location"],
                patient["symptoms"], risk_level, final_response, response_time,
            ),
        )


def get_history(limit=50):
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM consultations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM consultations")
