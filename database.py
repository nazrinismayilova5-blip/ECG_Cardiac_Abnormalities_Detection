import sqlite3

def init_db():

    conn = sqlite3.connect("patients.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        age INTEGER,
        sex TEXT,
        diagnosis TEXT,
        confidence REAL,
        risk TEXT,
        report_date TEXT
    )
    """)

    cur.execute("SELECT COUNT(*) FROM reports")
    count = cur.fetchone()[0]

    if count == 0:

        seed_data = [
        ("PT001",22,"Female","NORM",94.1,"Low","2026-04-25"),
        ("PT001",22,"Female","STTC",81.3,"Moderate","2026-04-20"),
        ("PT001",22,"Female","NORM",96.7,"Low","2026-04-15")
        ]

        cur.executemany("""
        INSERT INTO reports(
        patient_id,
        age,
        sex,
        diagnosis,
        confidence,
        risk,
        report_date
        )
        VALUES (?,?,?,?,?,?,?)
        """, seed_data)

    conn.commit()
    conn.close()