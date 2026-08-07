"""
PulseIQ Backend API
====================
Real FastAPI web service that loads the actual trained ML/DL models,
scaler, feature-selection lists and the held-out test dataset from
`ml_assets/` (copied directly from the user's project_2_complete_workspace),
and exposes them over HTTP so the PulseIQ frontend can display REAL
model predictions, REAL SHAP / Integrated-Gradients explanations, and
REAL clinically-validated metrics instead of hardcoded demo numbers.

Run with:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8001

All classical-ML and single-modality-DL numbers below are computed LIVE,
on server startup, against ml_assets/data/processed_test_unseen.npz
(11-subject / 327-window held-out test set the models never trained on).

The calibrated-DL, 15-min-recalibrated-DL and multimodal (ECG+PPG+PTT)
rows in /api/model-comparison are NOT re-computed live (those pipelines
require baseline-anchoring / recalibration / ECG-PTT extraction steps
that are out of scope for this API) -- they are reported verbatim from
the project's own markdown evaluation reports, and are clearly flagged
with "source": "report" so the frontend never presents them as live.
The multimodal row is additionally flagged as research-only because its
PTT feature is derived from the true BP label during training (known
label-leakage) -- it must never be framed as a live/invertible endpoint.
"""

import os
import sys
import time
import json
import uuid
import sqlite3
import datetime
import warnings
import threading
from typing import Optional

import numpy as np
import joblib
import torch
import requests
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

import auth

warnings.filterwarnings("ignore")

# A free instance is one shared core and 512 MB. Torch defaults to one thread
# per core and each pool costs memory it will never earn back here.
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml_assets")
SRC_DIR = os.path.join(ML_DIR, "src")
MODELS_DIR = os.path.join(ML_DIR, "models")
DATA_DIR = os.path.join(ML_DIR, "data")
# On a hosted deployment the app directory is read-only (and rebuilt on every
# push), so writable state lives wherever PULSEIQ_STATE_DIR points. Locally it
# defaults to the app directory, which is exactly the old behaviour.
STATE_DIR = os.environ.get("PULSEIQ_STATE_DIR", BASE_DIR)
os.makedirs(STATE_DIR, exist_ok=True)
VAULT_DIR = os.path.join(STATE_DIR, "vault_files")   # real uploaded documents

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, ML_DIR)

from feature_engineering import extract_window_features  # noqa: E402
from config import Config  # noqa: E402
from dl_model import PPGResNetBiLSTM  # noqa: E402
from metrics import (  # noqa: E402
    compute_regression_metrics,
    evaluate_aami_sp10,
    evaluate_bhs_grade,
)

TARGETS = ["SBP", "DBP", "MAP"]

# ---------------------------------------------------------------------------
# Demo patients -> real subject IDs in the held-out test set.
# These are the fictional names already used across the frontend
# (icu-monitor.html, patient-dashboard.html, doctor-dashboard.html).
# Each is pinned to a REAL subject_id from processed_test_unseen.npz so
# every prediction/explanation is computed from real PPG/VPG/APG windows,
# not synthetic data.
# ---------------------------------------------------------------------------
# EVERY unique subject in the test set gets a patient entry -- the pool is a
# name supply, not a limit. If the dataset ever contains more subjects than
# names here, the extras fall back to "Subject <id>" rather than being dropped,
# so no real subject is ever silently hidden from the roster.
DEMO_PATIENT_POOL = [
    ("rohit-sharma",    "Rohit Sharma"),
    ("meena-patil",     "Meena Patil"),
    ("arvind-kumar",    "Arvind Kumar"),
    ("sunita-nair",     "Sunita Nair"),
    ("imran-qureshi",   "Imran Qureshi"),
    ("lakshmi-iyer",    "Lakshmi Iyer"),
    ("david-fernandes", "David Fernandes"),
    ("neha-bansal",     "Neha Bansal"),
    ("thomas-mathew",   "Thomas Mathew"),
    ("priya-desai",     "Priya Desai"),
    ("aakash-rao",      "Aakash Rao"),
    ("farah-siddiqui",  "Farah Siddiqui"),
]

STATE = {}

# ---------------------------------------------------------------------------
# Health chatbot -- backed by a LOCAL LLM via Ollama (free, no API key,
# runs entirely on your own machine at http://localhost:11434). Nothing here
# calls Claude, OpenAI, or any hosted API -- the model referenced below must
# be pulled locally first with `ollama pull <model>`. See README.md.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Chat backend. Two real providers, chosen by which credentials exist:
#   * Groq  -- free hosted API, used when GROQ_API_KEY is set (deployments).
#   * Ollama -- local model, used otherwise (development machine).
# The key is only ever read from the environment; it is never written to disk
# or returned to the client.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = 120

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = 60


def chat_provider() -> str:
    return "groq" if GROQ_API_KEY else "ollama"

CHAT_SYSTEM_PROMPT = (
    "You are the PulseIQ Health Assistant, a general wellness chatbot inside a "
    "cuffless blood-pressure monitoring app. Rules you must always follow: "
    "1) Give general, evidence-based health/wellness information only -- never "
    "diagnose a condition, never tell the user to change or stop a medication "
    "dose, never claim to interpret their specific lab or clinical results as "
    "a diagnosis. "
    "2) If the user describes symptoms that could be an emergency (e.g. chest "
    "pain, severe shortness of breath, signs of stroke), tell them clearly to "
    "seek emergency care immediately. "
    "3) Keep answers concise and practical (roughly 2-5 sentences unless the "
    "user clearly wants more detail). "
    "4) Always keep in mind you are not a substitute for professional medical "
    "advice -- the app already shows a persistent disclaimer to the user, so "
    "you do not need to repeat it in every message, but do mention consulting "
    "a doctor when the topic is genuinely clinical (new symptoms, medication "
    "questions, abnormal readings). "
    "5) If BP/vitals context is provided below, you may reference it naturally, "
    "but do not overstate what a single reading means."
)


def _load_classical():
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler_StandardScaler.joblib"))
    feature_names = list(scaler.feature_names_in_)
    selected = {
        t: list(joblib.load(os.path.join(MODELS_DIR, f"selected_features_{t}.joblib")))
        for t in TARGETS
    }
    models = {
        t: joblib.load(os.path.join(MODELS_DIR, f"best_model_{t}.joblib"))
        for t in TARGETS
    }
    return scaler, feature_names, selected, models


def _load_dl():
    model = PPGResNetBiLSTM()
    state_dict = torch.load(
        os.path.join(MODELS_DIR, "best_resnet_bilstm_model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _extract_features_row(ppg, vpg, apg):
    """
    Feature vector in the exact column order the scaler was fitted on.

    This used to build a one-row DataFrame and index it by column name. The
    result is identical -- scaler.feature_names_in_ IS that column order -- but
    a plain array avoids importing pandas, which costs ~60 MB of resident
    memory for this single call and matters on a small instance.
    """
    feats = extract_window_features(ppg, vpg, apg, Config.SAMPLING_RATE)
    names = STATE["feature_names"]
    return np.array([[float(feats[n]) for n in names]], dtype=np.float64)


def classical_predict_window(i: int):
    """Real AdaBoost/SVR predictions for test-set window i, per target."""
    X = STATE["X"]
    row = _extract_features_row(X[i, 0, :], X[i, 1, :], X[i, 2, :])
    Xs = STATE["scaler"].transform(row)
    preds = {}
    for t in TARGETS:
        idx = [STATE["feature_names"].index(f) for f in STATE["selected"][t]]
        preds[t] = float(STATE["classical_models"][t].predict(Xs[:, idx])[0])
    return preds, Xs


def dl_predict_window(i: int):
    X = STATE["X"]
    x = torch.tensor(X[i : i + 1], dtype=torch.float32)
    with torch.no_grad():
        pred = STATE["dl_model"](x).numpy()[0]
    return {t: float(pred[j]) for j, t in enumerate(TARGETS)}


def resolve_patient(patient_id: str) -> int:
    """Return the first test-set window index belonging to a demo patient."""
    if patient_id not in STATE["patient_index"]:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    return STATE["patient_index"][patient_id]["first_window"]


def lookup_patient(key: str) -> Optional[dict]:
    """
    Accept either a demo key ("rohit-sharma") or a raw dataset subject id
    ("p119"), so a real account linked to a recording can address it directly
    without going through the demo naming layer.
    """
    idx = STATE.get("patient_index", {})
    if key in idx:
        return idx[key]
    for pid, info in idx.items():
        if info["subject_id"] == key:
            return info
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="PulseIQ Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    t0 = time.time()
    print("[PulseIQ] Initializing auth database (pulseiq.db) ...")
    auth.init_db()
    auth.init_feature_tables()
    os.makedirs(VAULT_DIR, exist_ok=True)

    print("[PulseIQ] Loading dataset ...")
    npz = np.load(os.path.join(DATA_DIR, "processed_test_unseen.npz"))
    X, Y, subject_ids = npz["X"], npz["Y"], npz["subject_ids"]
    STATE["X"] = X
    STATE["Y"] = Y
    STATE["subject_ids"] = subject_ids

    print("[PulseIQ] Loading classical ML models ...")
    scaler, feature_names, selected, classical_models = _load_classical()
    STATE["scaler"] = scaler
    STATE["feature_names"] = feature_names
    STATE["selected"] = selected
    STATE["classical_models"] = classical_models

    print("[PulseIQ] Loading deep-learning model (PPGResNetBiLSTM) ...")
    STATE["dl_model"] = _load_dl()

    print("[PulseIQ] Mapping demo patients to real subject IDs ...")
    unique_subjects = list(dict.fromkeys(subject_ids.tolist()))
    patient_index = {}
    for i, subj in enumerate(unique_subjects):          # EVERY subject, not a fixed 4
        if i < len(DEMO_PATIENT_POOL):
            pid, real_name = DEMO_PATIENT_POOL[i]
        else:
            pid, real_name = f"subject-{subj}", f"Subject {subj}"
        window_idxs = np.where(subject_ids == subj)[0].tolist()
        patient_index[pid] = {
            "name": real_name,
            "subject_id": str(subj),
            "bed": f"{i + 1:02d}",
            "display_id": f"P2-2024-{i + 1:03d}",
            "first_window": int(window_idxs[0]),
            "window_indices": window_idxs,
            "num_windows": len(window_idxs),
        }
    STATE["patient_index"] = patient_index

    # Everything above is fast. What follows -- scoring all 327 held-out windows
    # and building the SHAP background -- is the expensive part, and on a small
    # shared-CPU instance it can take minutes. Running it inline would keep the
    # server from binding its port, and the host would kill the deploy as
    # unresponsive. So it runs on a background thread: the API answers requests
    # immediately, and the few endpoints that need these numbers report
    # "computing" until they are ready.
    STATE["warmup"] = {"state": "running", "startedAt": time.time()}

    def _warmup():
        w0 = time.time()
        try:
            _compute_live_metrics(X, Y, unique_subjects)
            _build_shap_background(X, feature_names, selected)
            STATE["warmup"] = {"state": "ready", "seconds": round(time.time() - w0, 1)}
            print(f"[PulseIQ] Warm-up complete in {time.time() - w0:.1f}s.")
        except Exception as e:                       # never kill the server over this
            STATE["warmup"] = {"state": "failed", "error": str(e)}
            print(f"[PulseIQ] Warm-up FAILED: {e}")

    threading.Thread(target=_warmup, name="pulseiq-warmup", daemon=True).start()
    print(f"[PulseIQ] Serving after {time.time() - t0:.1f}s; "
          f"metrics warming up in the background.")


def _compute_live_metrics(X, Y, unique_subjects):
    print("[PulseIQ] Computing LIVE metrics over full held-out test set "
          f"({X.shape[0]} windows, {len(unique_subjects)} subjects) ...")
    classical_preds = np.zeros_like(Y)
    dl_preds = np.zeros_like(Y)
    for i in range(X.shape[0]):
        c_pred, _ = classical_predict_window(i)
        d_pred = dl_predict_window(i)
        for j, t in enumerate(TARGETS):
            classical_preds[i, j] = c_pred[t]
            dl_preds[i, j] = d_pred[t]
    STATE["classical_test_preds"] = classical_preds
    STATE["dl_test_preds"] = dl_preds

    live_comparison = {}
    for j, t in enumerate(TARGETS):
        y_true = Y[:, j]
        c_metrics = compute_regression_metrics(y_true, classical_preds[:, j])
        c_aami = evaluate_aami_sp10(y_true, classical_preds[:, j], t)
        c_bhs = evaluate_bhs_grade(y_true, classical_preds[:, j], t)
        d_metrics = compute_regression_metrics(y_true, dl_preds[:, j])
        d_aami = evaluate_aami_sp10(y_true, dl_preds[:, j], t)
        d_bhs = evaluate_bhs_grade(y_true, dl_preds[:, j], t)
        live_comparison[t] = {
            "classical": {"metrics": c_metrics, "aami": c_aami, "bhs": c_bhs},
            "dl": {"metrics": d_metrics, "aami": d_aami, "bhs": d_bhs},
        }
    STATE["live_comparison"] = live_comparison


def _build_shap_background(X, feature_names, selected):
    # Small background sample for SHAP KernelExplainer (per-target, in the
    # model's own selected-feature space) -- built once so
    # /api/explain/shap/* doesn't pay this cost on every request.
    print("[PulseIQ] Building SHAP background samples ...")
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X.shape[0], size=min(20, X.shape[0]), replace=False)
    bg_features = {}
    for t in TARGETS:
        idx = [feature_names.index(f) for f in selected[t]]
        rows = []
        for i in bg_idx:
            _, Xs = classical_predict_window(int(i))
            rows.append(Xs[0, idx])
        bg_features[t] = np.array(rows)
    STATE["shap_background"] = bg_features


# ---------------------------------------------------------------------------
# Auth (real accounts, SQLite + hashed passwords + JWT sessions)
# ---------------------------------------------------------------------------
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str  # "patient" or "doctor"


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/signup")
def signup(req: SignupRequest):
    if "@" not in req.email or "." not in req.email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    try:
        user = auth.create_user(req.name, req.email, req.password, req.role)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = auth.create_access_token(user)
    return {"token": token, "user": user}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = auth.verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = auth.create_access_token(user)
    return {"token": token, "user": user}


@app.get("/api/auth/me")
def me(current_user: dict = Depends(auth.get_current_user)):
    return {"user": current_user}


# ---------------------------------------------------------------------------
# Linking an account to a real recording in the dataset
# ---------------------------------------------------------------------------
# A freshly signed-up account has no recording of its own. Rather than quietly
# showing someone else's waveform under their name, the account stays unlinked
# until a Patient ID is entered, and the UI says so.
class LinkSubjectRequest(BaseModel):
    subjectId: Optional[str] = None   # null unlinks


@app.get("/api/recordings")
def recordings():
    """Patient IDs available to link, so the picker can't offer a bad value."""
    out = []
    for pid, info in STATE.get("patient_index", {}).items():
        out.append({
            "subjectId": info["subject_id"],
            "demoKey": pid,
            "demoName": info["name"],
            "numWindows": info["num_windows"],
        })
    out.sort(key=lambda r: r["subjectId"])
    return {"recordings": out}


@app.post("/api/me/subject")
def link_subject(req: LinkSubjectRequest, current_user: dict = Depends(auth.get_current_user)):
    if req.subjectId in (None, ""):
        auth.set_user_subject(current_user["id"], None)
        return {"subjectId": None, "linked": False}

    info = lookup_patient(req.subjectId)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"No recording found with Patient ID '{req.subjectId}'. "
                   f"Check the ID and try again.",
        )
    auth.set_user_subject(current_user["id"], info["subject_id"])
    return {
        "subjectId": info["subject_id"],
        "linked": True,
        "numWindows": info["num_windows"],
    }


# ---------------------------------------------------------------------------
# Medical Vault — real file storage, owned by the account
# ---------------------------------------------------------------------------
VAULT_CATEGORIES = ["reports", "prescriptions", "lab", "scans"]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
# Only formats a clinic would actually hand over. Blocking executables here
# matters because these files are served back out for download.
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".csv", ".doc", ".docx"}


@app.get("/api/vault")
def vault_list(current_user: dict = Depends(auth.get_current_user)):
    conn = auth.get_db()
    try:
        rows = conn.execute(
            """SELECT id, original_name, mime, size_bytes, category, uploaded_at
               FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC""",
            (current_user["id"],),
        ).fetchall()
        docs = [{
            "id": r["id"], "name": r["original_name"], "mime": r["mime"],
            "sizeBytes": r["size_bytes"], "category": r["category"],
            "uploadedAt": r["uploaded_at"],
        } for r in rows]
        return {"documents": docs, "categories": VAULT_CATEGORIES,
                "totalBytes": sum(d["sizeBytes"] for d in docs)}
    finally:
        conn.close()


@app.post("/api/vault/upload")
async def vault_upload(
    file: UploadFile = File(...),
    category: str = Form("reports"),
    current_user: dict = Depends(auth.get_current_user),
):
    if category not in VAULT_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category must be one of {VAULT_CATEGORIES}")

    original = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"'{ext or 'this file type'}' isn't allowed. Accepted: {', '.join(sorted(ALLOWED_EXT))}",
        )

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="That file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is larger than the 15 MB limit")

    user_dir = os.path.join(VAULT_DIR, str(current_user["id"]))
    os.makedirs(user_dir, exist_ok=True)
    stored = uuid.uuid4().hex + ext           # never trust the client's name on disk
    with open(os.path.join(user_dir, stored), "wb") as f:
        f.write(data)

    conn = auth.get_db()
    try:
        cur = conn.execute(
            """INSERT INTO documents (user_id, original_name, stored_name, mime, size_bytes, category, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (current_user["id"], original, stored, file.content_type, len(data), category, time.time()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": original, "sizeBytes": len(data), "category": category}
    finally:
        conn.close()


def _owned_document(doc_id: int, user_id: int):
    conn = auth.get_db()
    try:
        r = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="Document not found")
    # Ownership is checked server-side; a guessed id must not expose someone else's file.
    if r["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="That document belongs to another account")
    return r


@app.get("/api/vault/{doc_id}/download")
def vault_download(doc_id: int, current_user: dict = Depends(auth.get_current_user)):
    r = _owned_document(doc_id, current_user["id"])
    path = os.path.join(VAULT_DIR, str(r["user_id"]), r["stored_name"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing from storage")
    return FileResponse(path, filename=r["original_name"],
                        media_type=r["mime"] or "application/octet-stream")


@app.delete("/api/vault/{doc_id}")
def vault_delete(doc_id: int, current_user: dict = Depends(auth.get_current_user)):
    r = _owned_document(doc_id, current_user["id"])
    path = os.path.join(VAULT_DIR, str(r["user_id"]), r["stored_name"])
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass                                  # DB row still goes, file is orphaned at worst
    conn = auth.get_db()
    try:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    return {"deleted": doc_id}


# ---------------------------------------------------------------------------
# Medicine reminders — real schedules and real adherence
# ---------------------------------------------------------------------------
class MedicineRequest(BaseModel):
    name: str
    dosage: Optional[str] = ""
    times: List[str]                          # ["08:00", "20:00"]
    notes: Optional[str] = ""


def _today():
    return datetime.date.today().isoformat()


def _valid_time(t: str) -> bool:
    try:
        datetime.datetime.strptime(t, "%H:%M")
        return True
    except ValueError:
        return False


@app.get("/api/medicines")
def medicines_list(days: int = 30, day: Optional[str] = None,
                   current_user: dict = Depends(auth.get_current_user)):
    days = max(1, min(120, days))
    # `day` lets the calendar show and edit any date, not just today.
    selected = day or _today()
    try:
        sel_date = datetime.date.fromisoformat(selected)
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
    if sel_date > datetime.date.today():
        raise HTTPException(status_code=400, detail="Cannot log doses for a future date")

    conn = auth.get_db()
    try:
        meds = conn.execute(
            "SELECT * FROM medicines WHERE user_id = ? AND active = 1 ORDER BY created_at",
            (current_user["id"],),
        ).fetchall()
        taken_sel = {
            (r["medicine_id"], r["slot"])
            for r in conn.execute(
                "SELECT medicine_id, slot FROM medicine_logs WHERE user_id = ? AND day = ?",
                (current_user["id"], selected),
            ).fetchall()
        }

        out = []
        for m in meds:
            slots = json.loads(m["times"])
            created = datetime.date.fromtimestamp(m["created_at"])
            out.append({
                "id": m["id"], "name": m["name"], "dosage": m["dosage"] or "",
                "notes": m["notes"] or "", "times": slots,
                "takenOnDay": [s for s in slots if (m["id"], s) in taken_sel],
                # A medicine can't be logged before it was added.
                "scheduledOnDay": created <= sel_date,
                "createdAt": m["created_at"],
            })

        # Adherence over the window: doses actually logged vs doses scheduled
        # since each medicine was created (never counting days before it existed).
        start = (datetime.date.today() - datetime.timedelta(days=days - 1))
        expected = 0
        for m in meds:
            slots = json.loads(m["times"])
            created = datetime.date.fromtimestamp(m["created_at"])
            first = max(start, created)
            if first <= datetime.date.today():
                expected += ((datetime.date.today() - first).days + 1) * len(slots)
        logged = conn.execute(
            "SELECT COUNT(*) AS n FROM medicine_logs WHERE user_id = ? AND day >= ?",
            (current_user["id"], start.isoformat()),
        ).fetchone()["n"]

        by_day = {
            r["day"]: r["n"] for r in conn.execute(
                """SELECT day, COUNT(*) AS n FROM medicine_logs
                   WHERE user_id = ? AND day >= ? GROUP BY day""",
                (current_user["id"], start.isoformat()),
            ).fetchall()
        }
        history = []
        for i in range(days):
            d = (start + datetime.timedelta(days=i)).isoformat()
            history.append({"day": d, "taken": by_day.get(d, 0)})

        return {
            "medicines": out,
            "today": _today(),
            "selectedDay": selected,
            "adherence": {
                "windowDays": days,
                "expected": expected,
                "taken": int(logged),
                "percent": round(logged / expected * 100, 1) if expected else None,
            },
            "history": history,
        }
    finally:
        conn.close()


@app.post("/api/medicines")
def medicine_create(req: MedicineRequest, current_user: dict = Depends(auth.get_current_user)):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Medicine name is required")
    times = [t.strip() for t in req.times if t and t.strip()]
    if not times:
        raise HTTPException(status_code=400, detail="Add at least one reminder time")
    bad = [t for t in times if not _valid_time(t)]
    if bad:
        raise HTTPException(status_code=400, detail=f"Invalid time format: {', '.join(bad)} (use HH:MM)")
    times = sorted(set(times))

    conn = auth.get_db()
    try:
        cur = conn.execute(
            """INSERT INTO medicines (user_id, name, dosage, times, notes, active, created_at)
               VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (current_user["id"], name, (req.dosage or "").strip(),
             json.dumps(times), (req.notes or "").strip(), time.time()),
        )
        conn.commit()
        return {"id": cur.lastrowid, "name": name, "times": times}
    finally:
        conn.close()


def _owned_medicine(med_id: int, user_id: int):
    conn = auth.get_db()
    try:
        r = conn.execute("SELECT * FROM medicines WHERE id = ?", (med_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(status_code=404, detail="Medicine not found")
    if r["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="That medicine belongs to another account")
    return r


@app.delete("/api/medicines/{med_id}")
def medicine_delete(med_id: int, current_user: dict = Depends(auth.get_current_user)):
    _owned_medicine(med_id, current_user["id"])
    conn = auth.get_db()
    try:
        conn.execute("DELETE FROM medicine_logs WHERE medicine_id = ?", (med_id,))
        conn.execute("DELETE FROM medicines WHERE id = ?", (med_id,))
        conn.commit()
    finally:
        conn.close()
    return {"deleted": med_id}


class DoseRequest(BaseModel):
    slot: str
    day: Optional[str] = None
    taken: bool = True


@app.post("/api/medicines/{med_id}/dose")
def medicine_dose(med_id: int, req: DoseRequest, current_user: dict = Depends(auth.get_current_user)):
    m = _owned_medicine(med_id, current_user["id"])
    slots = json.loads(m["times"])
    if req.slot not in slots:
        raise HTTPException(status_code=400, detail=f"'{req.slot}' is not a scheduled time for this medicine")
    day = req.day or _today()

    conn = auth.get_db()
    try:
        if req.taken:
            try:
                conn.execute(
                    "INSERT INTO medicine_logs (medicine_id, user_id, day, slot, taken_at) VALUES (?, ?, ?, ?, ?)",
                    (med_id, current_user["id"], day, req.slot, time.time()),
                )
            except sqlite3.IntegrityError:
                pass                          # already marked; tapping twice is harmless
        else:
            conn.execute(
                "DELETE FROM medicine_logs WHERE medicine_id = ? AND day = ? AND slot = ?",
                (med_id, day, req.slot),
            )
        conn.commit()
    finally:
        conn.close()
    return {"medicineId": med_id, "day": day, "slot": req.slot, "taken": req.taken}


# ---------------------------------------------------------------------------
# Direct messaging — real accounts talking to each other
# ---------------------------------------------------------------------------
class SendMessageRequest(BaseModel):
    recipientId: int
    body: str


@app.get("/api/contacts")
def contacts(current_user: dict = Depends(auth.get_current_user)):
    """Who this user can message: doctors see patients, patients see doctors."""
    want = "patient" if current_user["role"] == "doctor" else "doctor"
    people = auth.list_users_by_role(want, exclude_id=current_user["id"])
    out = []
    for p in people:
        last = auth.last_message_with(current_user["id"], p["id"])
        out.append({
            "id": p["id"],
            "name": p["name"],
            "role": p["role"],
            "lastMessage": last["body"] if last else None,
            "lastAt": last["createdAt"] if last else None,
            "unread": auth.unread_count_from(current_user["id"], p["id"]),
        })
    # Unread first, then most recent activity.
    out.sort(key=lambda c: (-(1 if c["unread"] else 0), c["lastAt"] is None, -(c["lastAt"] or 0)))
    return {"contacts": out, "you": current_user, "totalUnread": auth.total_unread(current_user["id"])}


@app.get("/api/unread")
def unread(current_user: dict = Depends(auth.get_current_user)):
    """Cheap poll for the sidebar badge, so any page can show a new-message dot."""
    return {"totalUnread": auth.total_unread(current_user["id"])}


@app.get("/api/notifications")
def notifications(current_user: dict = Depends(auth.get_current_user)):
    """
    Real events for the signed-in account only. Every item is derived from
    something that actually happened: a message that arrived, a dose that is
    scheduled and not yet logged, an account with no recording attached, or the
    band the model's current reading falls into. Nothing here is generated to
    fill the list -- an account with nothing going on gets an empty list.
    """
    uid = current_user["id"]
    items = []
    now = datetime.datetime.now()

    # --- 1. Unread messages, grouped by who sent them ---------------------
    conn = auth.get_db()
    try:
        rows = conn.execute(
            """SELECT m.sender_id AS sid, u.name AS sname, u.role AS srole,
                      COUNT(*) AS n, MAX(m.created_at) AS last_at,
                      (SELECT body FROM messages m2
                        WHERE m2.sender_id = m.sender_id AND m2.recipient_id = ?
                          AND m2.read_at IS NULL
                        ORDER BY m2.created_at DESC LIMIT 1) AS last_body
                 FROM messages m JOIN users u ON u.id = m.sender_id
                WHERE m.recipient_id = ? AND m.read_at IS NULL
             GROUP BY m.sender_id
             ORDER BY last_at DESC""",
            (uid, uid),
        ).fetchall()
    finally:
        conn.close()

    for r in rows:
        who = ("Dr. " if r["srole"] == "doctor" else "") + r["sname"]
        items.append({
            "id": f"msg-{r['sid']}",
            "kind": "message",
            "icon": "\U0001f4ac",
            "title": f"{r['n']} new message{'s' if r['n'] > 1 else ''} from {who}",
            "body": (r["last_body"] or "")[:110],
            "href": "chat.html",
            "at": r["last_at"],
            "unread": True,
        })

    # --- 2. Medicine doses: scheduled today and not logged ----------------
    if current_user["role"] == "patient":
        today = _today()
        conn = auth.get_db()
        try:
            meds = conn.execute(
                "SELECT * FROM medicines WHERE user_id = ? AND active = 1", (uid,)
            ).fetchall()
            taken = {
                (x["medicine_id"], x["slot"])
                for x in conn.execute(
                    "SELECT medicine_id, slot FROM medicine_logs WHERE user_id = ? AND day = ?",
                    (uid, today),
                ).fetchall()
            }
        finally:
            conn.close()

        overdue, upcoming = [], []
        for m in meds:
            for slot in json.loads(m["times"]):
                if (m["id"], slot) in taken:
                    continue
                try:
                    hh, mm = [int(x) for x in slot.split(":")[:2]]
                except ValueError:
                    continue
                when = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                label = f"{m['name']}{(' ' + m['dosage']) if m['dosage'] else ''}"
                (overdue if when < now else upcoming).append((when, label, slot))

        overdue.sort(key=lambda x: x[0])
        upcoming.sort(key=lambda x: x[0])

        if overdue:
            names = ", ".join(sorted({o[1] for o in overdue}))[:110]
            items.append({
                "id": "med-overdue",
                "kind": "medicine",
                "icon": "\U0001f48a",
                "title": f"{len(overdue)} dose{'s' if len(overdue) > 1 else ''} not logged today",
                "body": names,
                "href": "medicine-reminders.html",
                "at": overdue[0][0].timestamp(),
                "unread": True,
            })
        if upcoming:
            when, label, slot = upcoming[0]
            items.append({
                "id": "med-next",
                "kind": "medicine",
                "icon": "\u23f0",
                "title": f"Next dose at {slot}",
                "body": label,
                "href": "medicine-reminders.html",
                "at": when.timestamp(),
                "unread": False,
            })

        # --- 3. Account with no recording attached ------------------------
        if not current_user.get("subject_id"):
            items.append({
                "id": "no-subject",
                "kind": "account",
                "icon": "\U0001f517",
                "title": "No recording linked to this account",
                "body": "Readings stay empty until a Patient ID is linked.",
                "href": "profile.html",
                "at": now.timestamp(),
                "unread": True,
            })
        else:
            # --- 4. What band the model's current reading falls into ------
            try:
                info = lookup_patient(current_user["subject_id"])
                if info is not None:
                    dl = dl_predict_window(info["first_window"])
                    cls = classify_bp(dl["SBP"], dl["DBP"])
                    if cls["band"] != "Normal":
                        items.append({
                            "id": "bp-band",
                            "kind": "reading",
                            "icon": "\u2764\ufe0f",
                            "title": f"Latest reading reads {cls['band']} "
                                     f"({dl['SBP']:.0f}/{dl['DBP']:.0f} mmHg)",
                            "body": cls["why"] + " - model estimate, not a diagnosis.",
                            "href": "xai-results.html",
                            "at": now.timestamp(),
                            "unread": True,
                        })
            except Exception:
                pass

    items.sort(key=lambda x: x.get("at") or 0, reverse=True)
    return {
        "items": items,
        "unreadCount": sum(1 for i in items if i.get("unread")),
        "generatedAt": now.timestamp(),
    }


@app.get("/api/messages/{other_id}")
def messages(other_id: int, current_user: dict = Depends(auth.get_current_user)):
    other = auth.get_user_by_id(other_id)
    if not other:
        raise HTTPException(status_code=404, detail="That user no longer exists")
    # Patients message doctors and vice versa — same-role chat isn't a thing here.
    if other["role"] == current_user["role"]:
        raise HTTPException(status_code=403, detail="You can only message the other role")
    # Opening the thread is what marks it read.
    marked = auth.mark_conversation_read(current_user["id"], other_id)
    return {
        "with": {"id": other["id"], "name": other["name"], "role": other["role"]},
        "messages": auth.get_conversation(current_user["id"], other_id),
        "markedRead": marked,
        "totalUnread": auth.total_unread(current_user["id"]),
    }


@app.post("/api/messages")
def send_message(req: SendMessageRequest, current_user: dict = Depends(auth.get_current_user)):
    other = auth.get_user_by_id(req.recipientId)
    if not other:
        raise HTTPException(status_code=404, detail="That user no longer exists")
    if other["role"] == current_user["role"]:
        raise HTTPException(status_code=403, detail="You can only message the other role")
    try:
        return auth.insert_message(current_user["id"], req.recipientId, req.body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "warmup": warmup_state(),
        "status": "ok",
        "models_loaded": {
            "classical": list(STATE.get("classical_models", {}).keys()),
            "dl": STATE.get("dl_model") is not None,
        },
        "test_set_windows": int(STATE["X"].shape[0]) if "X" in STATE else 0,
        "test_set_subjects": len(set(STATE["subject_ids"].tolist())) if "subject_ids" in STATE else 0,
    }


@app.get("/api/patients")
def list_patients():
    out = []
    for pid, info in STATE["patient_index"].items():
        out.append({
            "id": pid,
            "name": info["name"],
            "subjectId": info["subject_id"],
            "bed": info["bed"],
            "displayId": info["display_id"],
            "numWindows": info["num_windows"],
        })
    return {"patients": out}


def warmup_state() -> dict:
    return STATE.get("warmup", {"state": "unknown"})


def require_warm():
    """
    Guard for the routes that need the held-out metrics or the SHAP background.
    While the background warm-up is still running these genuinely do not exist
    yet, so the honest answer is 503 with a retry hint -- not a made-up number.
    """
    w = warmup_state()
    if w.get("state") == "ready":
        return
    if w.get("state") == "failed":
        raise HTTPException(
            status_code=503,
            detail=f"Model metrics could not be computed: {w.get('error')}")
    raise HTTPException(
        status_code=503,
        detail="Still warming up -- scoring the held-out test set. "
               "This takes up to a couple of minutes on a small instance. "
               "Retrying shortly will succeed.")


def prediction_reliability(classical_pred: dict, dl_pred: dict) -> dict:
    """
    Honest reliability figures for a single prediction. There is no calibrated
    per-sample confidence from this model, so nothing here is invented: every
    number is either the deep model's measured error on the held-out test set
    or a direct comparison of the two independent models on THIS window.
    """
    live = STATE.get("live_comparison") or {}
    out = {"warmingUp": warmup_state().get("state") == "running"}
    for t in TARGETS:
        entry = live.get(t, {}).get("dl", {})
        m = entry.get("metrics", {}) or {}
        bhs = entry.get("bhs", {}) or {}
        aami = entry.get("aami", {}) or {}
        c = classical_pred.get(t)
        d = dl_pred.get(t)
        out[t] = {
            # Mean absolute error of this model across every held-out window.
            "maeMmHg": m.get("MAE"),
            "rmseMmHg": m.get("RMSE"),
            # BHS cumulative bands: share of test predictions within N mmHg.
            "pctWithin5": bhs.get("pct_le_5mmHg"),
            "pctWithin10": bhs.get("pct_le_10mmHg"),
            "pctWithin15": bhs.get("pct_le_15mmHg"),
            "bhsGrade": bhs.get("bhs_grade"),
            "aamiPass": aami.get("aami_compliant"),
            # How far the two independent models land apart on this window.
            "modelGapMmHg": (abs(c - d) if (c is not None and d is not None) else None),
        }
    out["note"] = (
        "maeMmHg / pctWithin* are measured on the held-out test set, not a "
        "per-sample confidence. modelGapMmHg is the distance between the "
        "classical and deep models on this window."
    )
    return out


@app.get("/api/predict/{patient_id}")
def predict(patient_id: str, window: Optional[int] = None):
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    if window is not None:
        if window not in info["window_indices"]:
            raise HTTPException(status_code=400, detail="window index does not belong to this patient")
        i = window
    else:
        i = info["first_window"]

    classical_pred, _ = classical_predict_window(i)
    dl_pred = dl_predict_window(i)
    true_label = {t: float(STATE["Y"][i, j]) for j, t in enumerate(TARGETS)}

    return {
        "patientId": patient_id,
        "patientName": info["name"],
        "subjectId": info["subject_id"],
        "windowIndex": i,
        "numWindowsAvailable": info["num_windows"],
        "classical": classical_pred,
        "dl": dl_pred,
        "trueLabel": true_label,
        "reliability": prediction_reliability(classical_pred, dl_pred),
        "note": "classical = best per-target classical model (AdaBoost/SVR); "
                "dl = PPGResNetBiLSTM (single-modality). trueLabel is the "
                "ground-truth arterial-line BP for this held-out window.",
    }


@app.get("/api/waveform/{patient_id}")
def waveform(patient_id: str, window: Optional[int] = None, downsample: int = 3):
    """
    The real recorded PPG / VPG / APG waveform for this patient's real test
    subject -- the exact three channels the deep model consumes as input.
    Lightweight: pure array slicing, no model inference, so the ICU monitor
    can poll it cheaply. Heart rate is derived from real PPG systolic peaks.
    """
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    i = window if window is not None else info["first_window"]
    if i not in info["window_indices"]:
        raise HTTPException(status_code=400, detail="window index does not belong to this patient")

    downsample = max(1, min(10, downsample))
    X = STATE["X"]
    ppg = X[i, 0, :]
    fs = Config.SAMPLING_RATE

    # Real heart rate from systolic peaks (>=0.4s apart -> max 150 bpm)
    hr_bpm = None
    try:
        from scipy.signal import find_peaks
        norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        peaks, _ = find_peaks(norm, distance=int(0.4 * fs), height=0.2)
        if len(peaks) >= 2:
            hr_bpm = float(60.0 * fs / np.mean(np.diff(peaks)))
    except Exception:
        pass

    dl_pred = dl_predict_window(i)

    return {
        "patientId": patient_id,
        "patientName": info["name"],
        "subjectId": info["subject_id"],
        "windowIndex": int(i),
        "numWindowsAvailable": info["num_windows"],
        "windowIndices": info["window_indices"],
        "samplingRateHz": fs,
        "downsample": downsample,
        "durationSec": float(X.shape[2] / fs),
        "channels": {
            "ppg": X[i, 0, ::downsample].tolist(),
            "vpg": X[i, 1, ::downsample].tolist(),
            "apg": X[i, 2, ::downsample].tolist(),
        },
        "heartRateBpm": hr_bpm,
        "dl": dl_pred,
        "trueLabel": {t: float(STATE["Y"][i, j]) for j, t in enumerate(TARGETS)},
        "note": "PPG/VPG/APG are the real recorded channels for this subject. "
                "Heart rate is computed from real PPG peaks. SpO2 and "
                "respiration are NOT in this dataset and are not returned.",
    }


@app.get("/api/subject/{patient_id}")
def subject_detail(patient_id: str):
    """
    Full transparency on the REAL person behind a demo patient: every real
    window belonging to their real subject_id in processed_test_unseen.npz,
    with real ground-truth BP and real model predictions for each one (from
    the cached full-test-set evaluation computed at startup -- no extra
    inference cost here). This is what a demo name like "Rohit Sharma"
    actually maps to underneath: a real, anonymized ICU subject_id, not a
    real named person.
    """
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")

    Y = STATE["Y"]
    classical_preds = STATE["classical_test_preds"]
    dl_preds = STATE["dl_test_preds"]

    windows = []
    for i in info["window_indices"]:
        windows.append({
            "windowIndex": int(i),
            "trueLabel": {t: float(Y[i, j]) for j, t in enumerate(TARGETS)},
            "classical": {t: float(classical_preds[i, j]) for j, t in enumerate(TARGETS)},
            "dl": {t: float(dl_preds[i, j]) for j, t in enumerate(TARGETS)},
        })

    dl_errors = {
        t: float(np.mean(np.abs(dl_preds[info["window_indices"], j] - Y[info["window_indices"], j])))
        for j, t in enumerate(TARGETS)
    }

    return {
        "patientId": patient_id,
        "displayName": info["name"],
        "realSubjectId": info["subject_id"],
        "totalRealWindows": info["num_windows"],
        "datasetSource": "processed_test_unseen.npz (held-out test set, unseen during training)",
        "perWindowMeanAbsError_DL": dl_errors,
        "windows": windows,
        "disclosure": f"'{info['name']}' is a fictional display name PulseIQ assigns to real "
                       f"anonymized subject '{info['subject_id']}' from the held-out test set, "
                       f"purely so the demo has a human-readable identity. Every reading, "
                       f"waveform, and prediction shown for this patient is computed from that "
                       f"real subject's real recorded data -- none of it is synthetic.",
    }


# ---------------------------------------------------------------------------
# Clinical explanation layer
# ---------------------------------------------------------------------------
# Raw feature names ("vpg_spectral_centroid") are meaningless to a clinician or
# a patient, so each engineered feature is mapped to the physiological property
# it actually measures, and grouped into themes a human can reason about.
# Descriptions state what the signal property IS; they deliberately avoid
# asserting causation about the person's health.
FEATURE_GROUPS = {
    "rate": {
        "label": "Heart rate & rhythm",
        "what": "How fast the heart is beating and how regular the beat-to-beat timing is.",
        "features": [
            "ppg_hr_bpm", "ppg_pulse_interval_std",
            "ppg_lf_hf_ratio", "vpg_lf_hf_ratio", "apg_lf_hf_ratio",
        ],
    },
    "shape": {
        "label": "Pulse shape & timing",
        "what": "How wide each pulse is and how quickly it rises and falls — narrower, faster pulses tend to accompany higher pressure.",
        "features": [
            "ppg_pulse_width_25", "ppg_pulse_width_50", "ppg_pulse_width_75",
            "ppg_pulse_width_90", "ppg_stat_kurtosis", "vpg_stat_kurtosis",
        ],
    },
    "stiffness": {
        "label": "Arterial stiffness indicators",
        "what": "Features of the pulse's acceleration wave that reflect how stiff or compliant the arteries appear.",
        "features": [
            "apg_dominant_frequency", "apg_spectral_centroid",
            "apg_spectral_entropy", "apg_spectral_rolloff",
        ],
    },
    "strength": {
        "label": "Pulse strength & perfusion",
        "what": "How strong the pulse signal is overall, reflecting blood volume reaching the sensor.",
        "features": [
            "ppg_signal_energy", "ppg_stat_p10", "ppg_stat_p75", "ppg_stat_p90",
            "ppg_stat_std", "ppg_stat_iqr", "ppg_stat_mad",
        ],
    },
    "complexity": {
        "label": "Waveform detail & regularity",
        "what": "How much fine structure and high-frequency detail the pulse waveform carries.",
        "features": [
            "ppg_spectral_centroid", "ppg_spectral_entropy", "ppg_spectral_rolloff",
            "ppg_band_energy_hf", "vpg_spectral_centroid", "vpg_spectral_entropy",
            "vpg_spectral_rolloff",
        ],
    },
}

FEATURE_LABELS = {
    "ppg_hr_bpm": "Heart rate",
    "ppg_pulse_interval_std": "Beat-to-beat timing variation",
    "ppg_lf_hf_ratio": "Autonomic balance (pulse)",
    "vpg_lf_hf_ratio": "Autonomic balance (upstroke)",
    "apg_lf_hf_ratio": "Autonomic balance (acceleration)",
    "ppg_pulse_width_25": "Pulse width (near peak)",
    "ppg_pulse_width_50": "Pulse width (mid height)",
    "ppg_pulse_width_75": "Pulse width (lower third)",
    "ppg_pulse_width_90": "Pulse width (near base)",
    "ppg_stat_kurtosis": "Pulse peak sharpness",
    "vpg_stat_kurtosis": "Upstroke sharpness",
    "apg_dominant_frequency": "Dominant stiffness frequency",
    "apg_spectral_centroid": "Acceleration wave balance",
    "apg_spectral_entropy": "Acceleration wave regularity",
    "apg_spectral_rolloff": "Acceleration high-frequency edge",
    "ppg_signal_energy": "Overall pulse strength",
    "ppg_stat_p10": "Pulse trough level",
    "ppg_stat_p75": "Upper pulse level",
    "ppg_stat_p90": "Pulse peak level",
    "ppg_stat_std": "Pulse amplitude spread",
    "ppg_stat_iqr": "Pulse amplitude range",
    "ppg_stat_mad": "Pulse amplitude deviation",
    "ppg_spectral_centroid": "Waveform frequency balance",
    "ppg_spectral_entropy": "Waveform regularity",
    "ppg_spectral_rolloff": "Waveform high-frequency edge",
    "ppg_band_energy_hf": "High-frequency pulse energy",
    "vpg_spectral_centroid": "Upstroke frequency balance",
    "vpg_spectral_entropy": "Upstroke regularity",
    "vpg_spectral_rolloff": "Upstroke high-frequency edge",
}

_GROUP_OF = {}
for _gk, _g in FEATURE_GROUPS.items():
    for _f in _g["features"]:
        _GROUP_OF[_f] = _gk


def classify_bp(sbp: float, dbp: float) -> dict:
    """Standard adult bands, graded on systolic; diastolic can only escalate.
    Describes the reading — explicitly not a diagnosis."""
    if sbp >= 140 or dbp >= 90:
        band, tone = "High", "high"
        why = f"systolic {sbp:.0f} is at or above 140, or diastolic {dbp:.0f} is at or above 90"
    elif sbp >= 130 or dbp >= 80:
        band, tone = "Elevated", "elevated"
        why = f"systolic {sbp:.0f} is at or above 130, or diastolic {dbp:.0f} is at or above 80"
    elif sbp >= 120:
        band, tone = "Borderline", "elevated"
        why = f"systolic {sbp:.0f} falls in the 120–129 range while diastolic stays under 80"
    elif sbp < 90:
        band, tone = "Low", "elevated"
        why = f"systolic {sbp:.0f} is below 90"
    else:
        band, tone = "Normal", "normal"
        why = f"systolic {sbp:.0f} is under 120 and diastolic {dbp:.0f} is under 80"
    return {
        "band": band,
        "tone": tone,
        "why": why,
        "bands": [
            {"name": "Low",        "range": "< 90",     "from": 60,  "to": 90},
            {"name": "Normal",     "range": "90–119",   "from": 90,  "to": 120},
            {"name": "Borderline", "range": "120–129",  "from": 120, "to": 130},
            {"name": "Elevated",   "range": "130–139",  "from": 130, "to": 140},
            {"name": "High",       "range": "≥ 140",    "from": 140, "to": 180},
        ],
    }


@app.get("/api/explain/summary/{patient_id}")
def explain_summary(patient_id: str, target: str = "SBP", window: Optional[int] = None):
    """
    Everything the 'why this reading' view needs, in clinical language:
    the prediction, how it's classified, and which physiological signal
    groups pushed the estimate up or down (real SHAP, grouped into themes).
    """
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    # The dataset and model are fixed, so this result is deterministic —
    # computing once and caching turns a ~15s wait into an instant response.
    cache_key = (patient_id, target, int(i))
    cached = STATE.setdefault("summary_cache", {}).get(cache_key)
    if cached is not None:
        return cached

    require_warm()
    import shap

    feature_names = STATE["feature_names"]
    selected = STATE["selected"][target]
    idx = [feature_names.index(f) for f in selected]

    classical_pred, Xs = classical_predict_window(i)
    dl_pred = dl_predict_window(i)

    instance = Xs[0, idx].reshape(1, -1)
    model = STATE["classical_models"][target]
    explainer = shap.Explainer(model.predict, STATE["shap_background"][target])
    sv = explainer(instance)
    values = sv.values[0]
    base_value = float(np.array(sv.base_values).ravel()[0])

    # Roll per-feature SHAP up into the clinical themes.
    group_totals = {k: 0.0 for k in FEATURE_GROUPS}
    group_feats = {k: [] for k in FEATURE_GROUPS}
    for f, v in zip(selected, values):
        gk = _GROUP_OF.get(f)
        if gk is None:
            continue
        group_totals[gk] += float(v)
        group_feats[gk].append({
            "feature": f,
            "label": FEATURE_LABELS.get(f, f),
            "shapValue": float(v),
        })

    groups = []
    for gk, g in FEATURE_GROUPS.items():
        if not group_feats[gk]:
            continue
        total = group_totals[gk]
        groups.append({
            "key": gk,
            "label": g["label"],
            "what": g["what"],
            "mmHg": round(total, 2),
            "direction": "raised" if total > 0 else ("lowered" if total < 0 else "neutral"),
            "topFeatures": sorted(group_feats[gk], key=lambda r: abs(r["shapValue"]), reverse=True)[:3],
        })
    groups.sort(key=lambda g: abs(g["mmHg"]), reverse=True)

    sbp, dbp = dl_pred["SBP"], dl_pred["DBP"]
    classification = classify_bp(sbp, dbp)

    result = {
        "patientId": patient_id,
        "patientName": info["name"],
        "windowIndex": int(i),
        "target": target,
        "reading": {
            "SBP": round(sbp, 1), "DBP": round(dbp, 1), "MAP": round(dl_pred["MAP"], 1),
        },
        "classification": classification,
        "magnitude": {
            "explainedModel": "Classical ML (AdaBoost/SVR)",
            "explainedTarget": target,
            "populationBaseline": round(base_value, 1),
            "modelEstimate": round(classical_pred[target], 1),
            "netShift": round(classical_pred[target] - base_value, 1),
            "groups": groups,
        },
        "deepModelEstimate": {k: round(v, 1) for k, v in dl_pred.items()},
        "note": "Group contributions are real SHAP values for the classical model on "
                "this window, rolled up by physiological theme. The headline reading is "
                "the deep model's estimate; both are shown so neither is implied to be "
                "the other. Descriptive only — not a diagnosis.",
    }
    STATE["summary_cache"][cache_key] = result
    return result


@app.get("/api/explain/shap/{patient_id}")
def explain_shap(patient_id: str, target: str = "SBP", window: Optional[int] = None):
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    require_warm()
    import shap  # imported lazily -- heavy import, only needed for this route

    feature_names = STATE["feature_names"]
    selected = STATE["selected"][target]
    idx = [feature_names.index(f) for f in selected]

    _, Xs = classical_predict_window(i)
    instance = Xs[0, idx].reshape(1, -1)
    background = STATE["shap_background"][target]

    model = STATE["classical_models"][target]
    explainer = shap.Explainer(model.predict, background)
    sv = explainer(instance)
    values = sv.values[0].tolist()

    contributions = sorted(
        [{"feature": f, "shapValue": v} for f, v in zip(selected, values)],
        key=lambda r: abs(r["shapValue"]),
        reverse=True,
    )

    return {
        "patientId": patient_id,
        "target": target,
        "windowIndex": i,
        "baseValue": float(np.array(sv.base_values).ravel()[0]),
        "prediction": float(model.predict(instance)[0]),
        "contributions": contributions,
    }


@app.get("/api/explain/saliency/{patient_id}")
def explain_saliency(patient_id: str, target: str = "SBP", window: Optional[int] = None):
    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")
    info = lookup_patient(patient_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{patient_id}'")
    i = window if window is not None else info["first_window"]

    from captum.attr import IntegratedGradients

    target_idx = TARGETS.index(target)
    model = STATE["dl_model"]
    x = torch.tensor(STATE["X"][i : i + 1], dtype=torch.float32, requires_grad=True)

    ig = IntegratedGradients(model)
    attributions = ig.attribute(x, target=target_idx, n_steps=32)
    attr = attributions.detach().numpy()[0]  # [3, 1500]

    with torch.no_grad():
        pred = model(x).numpy()[0]

    # Downsample to keep the payload light for the frontend chart (1500 -> 300 pts)
    step = 5
    return {
        "patientId": patient_id,
        "target": target,
        "windowIndex": i,
        "prediction": {t: float(pred[j]) for j, t in enumerate(TARGETS)},
        "samplingRateHz": Config.SAMPLING_RATE,
        "channels": {
            "ppg": STATE["X"][i, 0, ::step].tolist(),
            "vpg": STATE["X"][i, 1, ::step].tolist(),
            "apg": STATE["X"][i, 2, ::step].tolist(),
        },
        "attribution": {
            "ppg": attr[0, ::step].tolist(),
            "vpg": attr[1, ::step].tolist(),
            "apg": attr[2, ::step].tolist(),
        },
        "note": "Integrated Gradients attribution (captum), 32 steps, computed "
                "live against the real PPGResNetBiLSTM checkpoint.",
    }


# ---------------------------------------------------------------------------
# Reported (non-live) figures for model variants whose full pipeline
# (baseline-anchoring / periodic recalibration / ECG-PTT extraction) is out
# of scope for this API. Sourced verbatim from the project's own reports.
# ---------------------------------------------------------------------------
REPORTED_COMPARISON = {
    "calibrated_dl": {
        "label": "PPGResNetBiLSTM (Calibrated, 1-pt baseline anchor)",
        "source": "report:final_calibrated_dl_evaluation_report.md",
        "SBP": {"MAE": 20.07, "RMSE": 24.58, "aami": "FAIL", "bhsGrade": "D"},
        "DBP": {"MAE": 9.44, "RMSE": 11.97, "aami": "FAIL", "bhsGrade": "D"},
        "MAP": {"MAE": 9.09, "RMSE": 12.24, "aami": "FAIL", "bhsGrade": "D"},
    },
    "recalibrated_dl_15min": {
        "label": "PPGResNetBiLSTM (15-min Periodic Recalibration)",
        "source": "report:final_clinical_breakthrough_report.md",
        "SBP": {"MAE": 13.06, "RMSE": 15.94, "aami": "FAIL", "bhsGrade": "D"},
        "DBP": {"MAE": 6.21, "RMSE": 7.92, "aami": "PASS", "bhsGrade": "C"},
        "MAP": {"MAE": 7.05, "RMSE": 8.41, "aami": "PASS", "bhsGrade": "C"},
    },
    "multimodal_research": {
        "label": "Multimodal ECG+PPG+PTT (5-min Recalib) — RESEARCH ONLY",
        "source": "report:final_multimodal_spt_report.md",
        "researchOnly": True,
        "caveat": "PTT feature is derived from the true BP label during "
                  "training (known label leakage). Not a live/invertible "
                  "endpoint — shown for research comparison only.",
        "SBP": {"MAE": 3.23, "RMSE": 4.00, "aami": "PASS", "bhsGrade": "A"},
        "DBP": {"MAE": 4.65, "RMSE": 6.72, "aami": "PASS", "bhsGrade": "B"},
        "MAP": {"MAE": 3.91, "RMSE": 5.36, "aami": "PASS", "bhsGrade": "B"},
    },
}


@app.get("/api/model-comparison")
def model_comparison():
    require_warm()
    live = STATE["live_comparison"]
    rows = []

    for t in TARGETS:
        rows.append({
            "model": "Classical ML (best per-target: AdaBoost/SVR)",
            "target": t,
            "source": "live",
            "MAE": round(live[t]["classical"]["metrics"]["MAE"], 2),
            "RMSE": round(live[t]["classical"]["metrics"]["RMSE"], 2),
            "aami": "PASS" if live[t]["classical"]["aami"]["aami_compliant"] else "FAIL",
            "bhsGrade": live[t]["classical"]["bhs"]["bhs_grade"].replace("Grade ", ""),
        })
    for t in TARGETS:
        rows.append({
            "model": "PPGResNetBiLSTM (single-modality DL)",
            "target": t,
            "source": "live",
            "MAE": round(live[t]["dl"]["metrics"]["MAE"], 2),
            "RMSE": round(live[t]["dl"]["metrics"]["RMSE"], 2),
            "aami": "PASS" if live[t]["dl"]["aami"]["aami_compliant"] else "FAIL",
            "bhsGrade": live[t]["dl"]["bhs"]["bhs_grade"].replace("Grade ", ""),
        })
    for key, entry in REPORTED_COMPARISON.items():
        for t in TARGETS:
            rows.append({
                "model": entry["label"],
                "target": t,
                "source": entry["source"],
                "researchOnly": entry.get("researchOnly", False),
                "caveat": entry.get("caveat"),
                "MAE": entry[t]["MAE"],
                "RMSE": entry[t]["RMSE"],
                "aami": entry[t]["aami"],
                "bhsGrade": entry[t]["bhsGrade"],
            })

    return {
        "testSet": {
            "name": "processed_test_unseen.npz",
            "windows": int(STATE["X"].shape[0]),
            "subjects": len(set(STATE["subject_ids"].tolist())),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Health chatbot (local LLM via Ollama -- free, no API key, runs on your machine)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    patientId: Optional[str] = None  # optional: include recent BP as context


@app.get("/api/chat/health")
def chat_health():
    if chat_provider() == "groq":
        # A key exists; confirm it is actually accepted rather than assuming.
        try:
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=8,
            )
            r.raise_for_status()
            names = [m.get("id") for m in r.json().get("data", [])]
            return {
                "provider": "groq",
                "reachable": True,
                "configuredModel": GROQ_MODEL,
                "modelAvailable": GROQ_MODEL in names,
                "installedModels": names[:25],
            }
        except Exception as e:
            return {
                "provider": "groq",
                "reachable": False,
                "configuredModel": GROQ_MODEL,
                "error": str(e),
                "hint": "GROQ_API_KEY is set but the Groq API rejected it or "
                        "could not be reached. Check the key in your host's "
                        "secrets settings.",
            }
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", [])]
        return {
            "provider": "ollama",
            "reachable": True,
            "ollamaReachable": True,
            "configuredModel": OLLAMA_MODEL,
            "modelPulled": any(OLLAMA_MODEL in m for m in models),
            "installedModels": models,
        }
    except Exception as e:
        return {
            "provider": "ollama",
            "reachable": False,
            "ollamaReachable": False,
            "configuredModel": OLLAMA_MODEL,
            "error": str(e),
            "hint": ("The AI assistant is not configured on this server -- set a "
                     "GROQ_API_KEY (free from console.groq.com). Everything else "
                     "works without it.")
                    if os.environ.get("PORT") or os.environ.get("RENDER") else
                    ("No GROQ_API_KEY is set, so PulseIQ looked for a local Ollama "
                     "server. Install it from https://ollama.com/download, run "
                     f"'ollama pull {OLLAMA_MODEL}', and make sure it is running."),
        }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    system_prompt = CHAT_SYSTEM_PROMPT
    if req.patientId and req.patientId in STATE.get("patient_index", {}):
        try:
            info = STATE["patient_index"][req.patientId]
            i = info["first_window"]
            dl_pred = dl_predict_window(i)
            system_prompt += (
                f"\n\nContext: the user's most recent PulseIQ reading (from the "
                f"PPGResNetBiLSTM model) was approximately "
                f"{round(dl_pred['SBP'])}/{round(dl_pred['DBP'])} mmHg "
                f"(MAP {round(dl_pred['MAP'])}). Only mention this if it's "
                f"relevant to what they're asking."
            )
        except Exception:
            pass  # context is best-effort; never block the chat over it

    messages = [{"role": "system", "content": system_prompt}]
    for m in req.history[-10:]:  # keep recent context bounded
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    if chat_provider() == "groq":
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": messages,
                      "temperature": 0.4, "max_tokens": 700},
                timeout=GROQ_TIMEOUT_SECONDS,
            )
            if resp.status_code == 401:
                raise HTTPException(
                    status_code=502,
                    detail="The Groq API key was rejected. Check GROQ_API_KEY "
                           "in your host's secrets settings.")
            if resp.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Groq's free-tier rate limit was hit. Wait a moment "
                           "and send the message again.")
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            if not reply:
                raise ValueError("empty response from Groq")
            return {"reply": reply, "model": GROQ_MODEL, "source": "groq"}
        except HTTPException:
            raise
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="Groq took too long to respond.")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Groq error: {e}")

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("message", {}).get("content", "").strip()
        if not reply:
            raise ValueError("empty response from local model")
        return {"reply": reply, "model": OLLAMA_MODEL, "source": "local-llm"}
    except requests.exceptions.ConnectionError:
        # Two very different situations reach this line, so say which one it is
        # rather than telling a visitor on a hosted server to install Ollama.
        hosted = bool(os.environ.get("PORT") or os.environ.get("RENDER"))
        if hosted:
            detail = ("The AI assistant is not configured on this server. It needs a "
                      "GROQ_API_KEY environment variable (a free key from "
                      "console.groq.com). Everything else in PulseIQ works without it, "
                      "including messaging your doctor.")
        else:
            detail = (f"Local AI (Ollama) is not reachable at {OLLAMA_BASE_URL}. "
                      f"Install it from https://ollama.com/download, run "
                      f"'ollama pull {OLLAMA_MODEL}', and make sure it's running. "
                      f"Alternatively set GROQ_API_KEY to use the free hosted model.")
        raise HTTPException(status_code=503, detail=detail)
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Local AI took too long to respond. Try a smaller model "
                   "(e.g. 'ollama pull llama3.2:1b') if your machine is slow.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local AI error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", "8001")), reload=False)


# ---------------------------------------------------------------------------
# Serve the frontend from this same origin when a static/ folder is present.
# One origin means no CORS preflight and no mixed-content block, which is what
# makes the hosted build work over HTTPS. Mounted last so it can never shadow
# an /api/* route.
# ---------------------------------------------------------------------------
_STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(_STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
