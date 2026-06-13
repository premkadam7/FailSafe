"""
FAILSAFE — Early Student Failure Risk Detection System
Team: Prem Kadam · Bhavit · Krishna Jain | IIT Guwahati

Run locally:  uvicorn main:app --reload
Production:   uvicorn main:app --host 0.0.0.0 --port $PORT
Docs:         http://localhost:8000/docs
"""

import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="FAILSAFE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
MODEL_PATH = BASE / "failsafe_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\n\nfailsafe_model.pkl not found.\n"
        f"Place the pkl file in: {BASE}\n"
    )

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model         = bundle["model"]
encoders      = bundle["encoders"]
feature_names = bundle["feature_names"]
explainer     = bundle["explainer"]

print(f"[FAILSAFE] Model loaded — {len(feature_names)} features")

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = BASE / "failsafe.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    TEXT,
            risk_score    REAL,
            at_risk       INTEGER,
            interventions TEXT,
            shap_values   TEXT,
            features      TEXT,
            created_at    TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ── Intervention logic ────────────────────────────────────────────────────────

RULES: dict = {
    "absences":  lambda v: v > 6,
    "failures":  lambda v: v >= 1,
    "G1":        lambda v: v < 10,
    "G2":        lambda v: v < 10,
    "studytime": lambda v: v <= 1,
    "Dalc":      lambda v: v >= 3,
    "Walc":      lambda v: v >= 3,
    "health":    lambda v: v <= 2,
    "famrel":    lambda v: v <= 2,
    "goout":     lambda v: v >= 4,
}

MESSAGES: dict = {
    "absences":  "High absenteeism recorded — schedule a mandatory attendance review with the student's advisor.",
    "failures":  "One or more prior failures on record — assign a subject mentor and arrange remedial sessions.",
    "G1":        "Below-average Period 1 grade — initiate an early-intervention meeting with the class teacher.",
    "G2":        "Below-average Period 2 grade — enrol in supplementary tutoring before final assessments.",
    "studytime": "Very low reported study time — provide a structured weekly study plan.",
    "Dalc":      "Elevated weekday alcohol consumption — refer to the student counselling centre.",
    "Walc":      "Elevated weekend alcohol consumption — refer to the student counselling centre.",
    "health":    "Poor self-reported health — direct to campus medical services for an assessment.",
    "famrel":    "Strained family relations reported — connect with student welfare and support services.",
    "goout":     "Frequent social outings reducing study time — recommend a time-management workshop.",
}


def generate_interventions(student_row: np.ndarray, shap_row: np.ndarray, top_n: int = 5) -> list[str]:
    shap_df = pd.DataFrame(
        {"feature": feature_names, "shap": shap_row, "value": student_row}
    ).sort_values("shap", ascending=False)

    top_risk = shap_df[shap_df["shap"] > 0].head(top_n)
    plans: list[str] = []

    for _, row in top_risk.iterrows():
        feat, val = row["feature"], row["value"]
        if feat in RULES and RULES[feat](val):
            plans.append(MESSAGES[feat])

    return plans if plans else ["No critical risk factors identified. Continue regular academic monitoring."]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app")


@app.get("/app", include_in_schema=False)
def serve_frontend():
    html_path = BASE / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in server directory")
    return FileResponse(html_path)


@app.post("/upload", summary="Upload student CSV and run predictions")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()

    # Try semicolon-separated first (UCI format), then comma
    try:
        df = pd.read_csv(io.BytesIO(contents), sep=";")
        if df.shape[1] < 5:
            df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Drop target column if present — prevents leakage
    if "G3" in df.columns:
        df = df.drop(columns=["G3"])

    # Encode categorical columns using saved encoders
    df_enc = df.copy()
    for col, le in encoders.items():
        if col in df_enc.columns:
            try:
                df_enc[col] = le.transform(df_enc[col].astype(str))
            except Exception:
                df_enc[col] = 0

    # Fill any missing feature columns with 0
    for col in feature_names:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X     = df_enc[feature_names]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    sv    = explainer.shap_values(X)

    # Clear old batch and store new predictions
    conn = get_db()
    conn.execute("DELETE FROM predictions")

    results: list[dict] = []
    for i in range(len(df)):
        plans     = generate_interventions(X.values[i], sv[i])
        shap_dict = {k: float(v) for k, v in zip(feature_names, sv[i])}
        feat_dict = {k: float(v) for k, v in zip(feature_names, X.values[i])}
        sid       = f"S{i + 1:03d}"

        conn.execute(
            """INSERT INTO predictions
               (student_id, risk_score, at_risk, interventions, shap_values, features, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                sid,
                float(probs[i]),
                int(preds[i]),
                json.dumps(plans),
                json.dumps(shap_dict),
                json.dumps(feat_dict),
                datetime.now().isoformat(),
            ),
        )
        results.append(
            {
                "student_id":    sid,
                "risk_score":    round(float(probs[i]), 4),
                "at_risk":       bool(preds[i]),
                "interventions": plans,
            }
        )

    conn.commit()
    conn.close()

    scores    = [r["risk_score"] for r in results]
    at_risk_n = sum(1 for r in results if r["at_risk"])

    return {
        "total":     len(results),
        "at_risk":   at_risk_n,
        "safe":      len(results) - at_risk_n,
        "avg_risk":  round(float(np.mean(scores)), 4),
        "high_risk": sum(1 for s in scores if s >= 0.8),
        "med_risk":  sum(1 for s in scores if 0.5 <= s < 0.8),
        "low_risk":  sum(1 for s in scores if s < 0.5),
        "students":  results,
    }


@app.get("/students", summary="All predictions, sorted by risk score descending")
def get_students():
    conn = get_db()
    rows = conn.execute(
        "SELECT student_id, risk_score, at_risk, interventions FROM predictions ORDER BY risk_score DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/students/{student_id}", summary="Full detail for one student")
def get_student(student_id: str):
    conn = get_db()
    row  = conn.execute("SELECT * FROM predictions WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    r = dict(row)
    r["interventions"] = json.loads(r["interventions"])
    r["shap_values"]   = json.loads(r["shap_values"])
    r["features"]      = json.loads(r["features"])
    return r


@app.get("/summary", summary="Aggregate dashboard stats")
def get_summary():
    conn = get_db()
    rows = conn.execute("SELECT risk_score, at_risk FROM predictions").fetchall()
    conn.close()
    if not rows:
        return {"total": 0}
    scores    = [r["risk_score"] for r in rows]
    at_risk_n = sum(r["at_risk"] for r in rows)
    return {
        "total":     len(rows),
        "at_risk":   at_risk_n,
        "safe":      len(rows) - at_risk_n,
        "avg_risk":  round(sum(scores) / len(scores), 4),
        "high_risk": sum(1 for s in scores if s >= 0.8),
        "med_risk":  sum(1 for s in scores if 0.5 <= s < 0.8),
        "low_risk":  sum(1 for s in scores if s < 0.5),
    }
