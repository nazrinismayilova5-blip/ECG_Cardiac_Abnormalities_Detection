import numpy as np
import secrets
import sqlite3
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, File, Form, UploadFile, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from database import init_db
from controllers.application_controller import ApplicationController

load_dotenv()

SESSIONS: dict[str, dict] = {}
SESSION_TTL   = timedelta(hours=8)
SESSION_COOKIE = "cardioscan_session"

DEMO_EMAIL    = os.getenv("DEMO_EMAIL")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD")

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
RISK_MAP    = {
    "NORM": "Low",
    "MI":   "High",
    "STTC": "Moderate",
    "CD":   "Moderate",
    "HYP":  "Moderate",
}

app = FastAPI(title="CardioScan AI", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

controller = ApplicationController()
init_db()


def create_session(data: dict) -> str:
    token = secrets.token_hex(32)
    SESSIONS[token] = {"data": data, "expires": datetime.utcnow() + SESSION_TTL}
    return token


def get_session(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    entry = SESSIONS.get(token)
    if not entry:
        return None
    if datetime.utcnow() > entry["expires"]:
        SESSIONS.pop(token, None)
        return None
    return entry["data"]


def require_session(request: Request):
    session = get_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


@app.post("/api/login")
async def api_login(response: Response, email: str = Form(...), password: str = Form(...)):
    if email != DEMO_EMAIL or password != DEMO_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session({"email": email, "patient_id": "PT001"})
    response.set_cookie(
        key=SESSION_COOKIE, value=token,
        httponly=True, samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
    )
    return {"ok": True}


@app.post("/api/logout")
async def api_logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        SESSIONS.pop(token, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/api/me")
async def api_me(session: dict = Depends(require_session)):
    return session


@app.get("/")
def home():      return FileResponse("templates/home.html")

@app.get("/analyze")
def analyze():   return FileResponse("templates/analyze.html")

@app.get("/results")
def results():   return FileResponse("templates/results.html")

@app.get("/login")
def login():     return FileResponse("templates/login.html")

@app.get("/dashboard")
def dashboard(): return FileResponse("templates/dashboard.html")

@app.get("/reports")
def reports():   return FileResponse("templates/reports.html")


@app.get("/patient_history")
def patient_history(session: dict = Depends(require_session)):
    conn = sqlite3.connect("patients.db")
    cur  = conn.cursor()
    rows = cur.execute("""
        SELECT patient_id, diagnosis, confidence, risk, report_date
        FROM reports ORDER BY id DESC
    """).fetchall()
    conn.close()
    return [
        {"id": r[0], "diagnosis": r[1], "confidence": r[2], "risk": r[3], "date": r[4]}
        for r in rows
    ]


@app.post("/predict_full")
async def predict_full(
    request:  Request,
    ecg_file: UploadFile = File(...),
    age:      int        = Form(...),
    sex:      int        = Form(...),
    height:   float      = Form(...),
    weight:   float      = Form(...),
    text:     str        = Form(""),
):
    metadata = {"age": age, "sex": sex, "height": height, "weight": weight}

    result     = controller.predict(ecg_file=ecg_file, metadata=metadata, text=text)
    probs      = result[0]
    pred_class = CLASS_NAMES[int(np.argmax(probs))]
    risk       = RISK_MAP.get(pred_class, "Moderate")
    confidence = float(np.max(probs)) * 100

    conn = sqlite3.connect("patients.db")
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO reports(patient_id, age, sex, diagnosis, confidence, risk, report_date)
        VALUES(?, ?, ?, ?, ?, ?, DATE('now'))
    """, ("Test Patient", age, "Female" if sex == 0 else "Male",
          pred_class, round(confidence, 2), risk))
    conn.commit()
    conn.close()

    return {
        "predicted_class": pred_class,
        "probabilities":   probs,
        "risk":            risk,
        "confidence":      round(confidence, 1),
    }
