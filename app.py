import os
import shutil
import threading
import time
import webbrowser
from datetime import date as date_cls
from datetime import datetime, time as dtime, timedelta
from typing import Iterable, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from anti_static_images.liveness_model import LivenessModel
from attendance_manager import AttendanceEvent, AttendanceManager
from config import (
    ADMIN_PASSWORD as DEFAULT_ADMIN_PASSWORD,
    ADMIN_SESSION_SECONDS,
    CAM_DEFAULT_DROID_URL,
    DATASET_DIR,
    FACE_CAPTURE,
    MAX_IMAGES,
    OVERLAY_DURATION as DEFAULT_OVERLAY_DURATION,
    SHIFT_END_TIME,
    SHIFT_START_TIME,
    STRANGER_OVERLAY_DURATION,
)
from database import SessionLocal, init_db
from face_model import FaceModel
from models import AuditLog, AttendanceRow, Setting, User
from services.security import AdminSessionManager, PasswordHasher
from utils import face_cascade, preprocess_face, vn_no_diacritics


init_db()

app = FastAPI(title="Face Attendance System")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

camera_source: int | str = 0
cap = None
camera_active = False
camera_error = ""
latest_frame_jpeg = None
latest_frame_lock = threading.RLock()
CAMERA_MODE = "attendance"

last_event_text = ""
last_event_ts = 0.0
last_stranger_ts = 0.0

OVERLAY_DURATION = DEFAULT_OVERLAY_DURATION
ADMIN_PASSWORD = DEFAULT_ADMIN_PASSWORD
LIVENESS_BLOCK_PROB = 0.4

face_model = FaceModel()
face_model.load_or_train_on_start()
att_manager = AttendanceManager()
liveness_model = LivenessModel()
session_manager = AdminSessionManager(max_age_seconds=ADMIN_SESSION_SECONDS)

threshold = getattr(face_model, "threshold", 400.0)
label_map = getattr(face_model, "label_map", {})


def draw_corner_box(img, x, y, w, h, color, thickness=2, length=20):
    x2, y2 = x + w, y + h
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    cv2.line(img, (x2, y), (x2 - length, y), color, thickness)
    cv2.line(img, (x2, y), (x2, y + length), color, thickness)
    cv2.line(img, (x, y2), (x + length, y2), color, thickness)
    cv2.line(img, (x, y2), (x, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def parse_hhmm(value: str) -> dtime:
    try:
        hour, minute = map(int, value.split(":"))
        return dtime(hour=hour, minute=minute)
    except Exception:
        return dtime(hour=9, minute=0)


def normalize_camera_source(value: str | int | None) -> str | int:
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return str(value)


def reload_face_runtime() -> None:
    global threshold, label_map

    threshold = getattr(face_model, "threshold", 400.0)
    label_map = getattr(face_model, "label_map", {})


def get_or_create_settings(db) -> Setting:
    settings = db.query(Setting).first()
    if settings:
        if not settings.admin_password:
            settings.admin_password = PasswordHasher.hash(DEFAULT_ADMIN_PASSWORD)
            db.commit()
            db.refresh(settings)
        return settings

    settings = Setting(
        shift_start=SHIFT_START_TIME,
        shift_end=SHIFT_END_TIME,
        min_in_to_out=60,
        min_out_to_in=60,
        overlay_duration=DEFAULT_OVERLAY_DURATION,
        camera_source="0",
        admin_password=PasswordHasher.hash(DEFAULT_ADMIN_PASSWORD),
    )
    db.add(settings)
    db.commit()
    db.refresh(settings)
    return settings


def record_audit(
    action: str,
    entity: str | None = None,
    entity_id: str | int | None = None,
    detail: str | None = None,
    success: bool = True,
    actor: str = "admin",
) -> None:
    db = SessionLocal()
    try:
        db.add(AuditLog(
            actor=actor,
            action=action,
            entity=entity,
            entity_id=str(entity_id) if entity_id is not None else None,
            detail=detail,
            success=success,
        ))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def load_runtime_settings() -> None:
    global ADMIN_PASSWORD, OVERLAY_DURATION, camera_source

    db = SessionLocal()
    try:
        settings = get_or_create_settings(db)
        ADMIN_PASSWORD = settings.admin_password
        OVERLAY_DURATION = int(settings.overlay_duration or DEFAULT_OVERLAY_DURATION)
        camera_source = normalize_camera_source(settings.camera_source)
        att_manager.configure(
            shift_start=settings.shift_start or SHIFT_START_TIME,
            shift_end=settings.shift_end or SHIFT_END_TIME,
            min_in_to_out=settings.min_in_to_out or 60,
            min_out_to_in=settings.min_out_to_in or 60,
        )
    finally:
        db.close()


def is_admin(request: Request) -> bool:
    return session_manager.verify_token(request.cookies.get("admin_token"))


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="Admin authentication required")


def init_default_admin() -> None:
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if admin:
            return

        admin = User(name="Admin", role="admin")
        db.add(admin)
        db.commit()
        db.refresh(admin)

        admin.face_dir = os.path.join(DATASET_DIR, str(admin.id))
        os.makedirs(admin.face_dir, exist_ok=True)
        db.commit()
    finally:
        db.close()


def ensure_not_last_admin(db, user: User, new_role: str) -> None:
    if user.role == "admin" and new_role != "admin":
        count = db.query(User).filter(User.role == "admin").count()
        if count <= 1:
            raise HTTPException(status_code=400, detail="At least one administrator is required")


def day_bounds(day: date_cls) -> tuple[datetime, datetime]:
    start = datetime.combine(day, dtime.min)
    end = datetime.combine(day + timedelta(days=1), dtime.min)
    return start, end


def date_range(start: date_cls, end: date_cls) -> Iterable[date_cls]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_user_day_events(db, user_id: int, day: date_cls) -> list[AttendanceEvent]:
    start_dt, end_dt = day_bounds(day)
    rows = (
        db.query(AttendanceRow)
        .filter(
            AttendanceRow.user_id == user_id,
            AttendanceRow.ts >= start_dt,
            AttendanceRow.ts < end_dt,
        )
        .order_by(AttendanceRow.ts.asc())
        .all()
    )
    return [AttendanceEvent.from_row(row) for row in rows]


def record_attendance(user_id: int, name: str):
    db = SessionLocal()
    try:
        events = load_user_day_events(db, user_id, date_cls.today())
        decision = att_manager.handle(name, events=events, identity_key=user_id)
        if decision.accepted and decision.action:
            db.add(AttendanceRow(
                user_id=user_id,
                name=name,
                action=decision.action.lower(),
                ts=decision.timestamp,
                raw_time=decision.timestamp.strftime("%H:%M:%S"),
            ))
            db.commit()
        return decision
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def find_user(user_id: int | None) -> User | None:
    if user_id is None:
        return None

    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == int(user_id)).first()
    finally:
        db.close()


def update_event_overlay_from_manager() -> None:
    global last_event_text, last_event_ts

    event = att_manager.last_event
    if event and att_manager.last_event_ts > last_event_ts and event.get("text"):
        last_event_text = event["text"]
        last_event_ts = att_manager.last_event_ts


def render_event_overlay(frame) -> None:
    global last_event_text

    now = time.time()
    if not last_event_text or (now - last_event_ts) > OVERLAY_DURATION:
        last_event_text = ""
        return

    event = att_manager.last_event or {}
    level = event.get("level", "success")
    if level == "warning":
        color = (0, 215, 255)
        scale = 0.7
        thickness = 2
        shadow_thickness = 3
    elif level == "error":
        color = (0, 0, 255)
        scale = 0.7
        thickness = 2
        shadow_thickness = 3
    else:
        color = (0, 255, 0)
        scale = 0.9
        thickness = 2
        shadow_thickness = 3

    x_text = int(frame.shape[1] * 0.05)
    y_text = 50
    for line in last_event_text.split("\n"):
        cv2.putText(frame, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), shadow_thickness)
        cv2.putText(frame, line, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y_text += int(34 * scale)


def process_attendance_frame(frame) -> None:
    global last_stranger_ts

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(100, 100),
        maxSize=(400, 400),
    )

    update_event_overlay_from_manager()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_p = preprocess_face(roi_gray)
        face_color = frame[y:y + h, x:x + w]

        if not (face_model and getattr(face_model, "recognizer", None) is not None and face_model.has_model()):
            draw_corner_box(frame, x, y, w, h, (255, 255, 0))
            continue

        if liveness_model and liveness_model.has_model():
            liveness = liveness_model.predict(face_color)
            if liveness and not liveness["is_live"] and liveness["score"] >= LIVENESS_BLOCK_PROB:
                draw_corner_box(frame, x, y, w, h, (0, 140, 255))
                continue

        try:
            predicted_id, confidence = face_model.recognizer.predict(roi_p)
        except Exception:
            draw_corner_box(frame, x, y, w, h, (255, 255, 0))
            continue

        if confidence < threshold * 0.9:
            user = find_user(face_model.label_map.get(predicted_id))
            if not user:
                draw_corner_box(frame, x, y, w, h, (0, 0, 255))
                last_stranger_ts = time.time()
                continue

            decision = record_attendance(user.id, user.name)
            color = (0, 255, 0) if decision.accepted else (0, 215, 255)
            display_name = vn_no_diacritics(user.name)
            draw_corner_box(frame, x, y, w, h, color)
            cv2.putText(frame, display_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        elif confidence >= threshold * 1.1:
            draw_corner_box(frame, x, y, w, h, (0, 0, 255))
            last_stranger_ts = time.time()

    now = time.time()
    if last_stranger_ts and (now - last_stranger_ts) <= STRANGER_OVERLAY_DURATION:
        h_img, w_img = frame.shape[:2]
        cv2.putText(frame, "Stranger!", (w_img // 2 - 100, h_img // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    render_event_overlay(frame)


def process_capture_frame(frame) -> None:
    global CAMERA_MODE

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    valid = True

    if not FACE_CAPTURE.get("active"):
        CAMERA_MODE = "attendance"
        return

    if len(faces) != 1:
        FACE_CAPTURE["message"] = "Please keep exactly ONE face in the frame"
        valid = False

    if valid:
        x, y, w, h = faces[0]
        if h < frame.shape[0] // 5:
            FACE_CAPTURE["message"] = "Move closer to the camera"
            valid = False

    if valid:
        roi_gray = gray[y:y + h, x:x + w]
        roi_p = preprocess_face(roi_gray)
        face_color = frame[y:y + h, x:x + w]

        if liveness_model and liveness_model.has_model():
            liveness = liveness_model.predict(face_color)
            if liveness and not liveness["is_live"] and liveness["score"] >= LIVENESS_BLOCK_PROB:
                FACE_CAPTURE["message"] = "Liveness check failed"
                valid = False

    if not valid:
        return

    FACE_CAPTURE["frames"].append(roi_p.copy())
    FACE_CAPTURE["count"] += 1
    FACE_CAPTURE["progress"] = int(FACE_CAPTURE["count"] / MAX_IMAGES * 100)
    FACE_CAPTURE["message"] = f"Capturing: {FACE_CAPTURE['progress']}%"

    if FACE_CAPTURE["count"] >= MAX_IMAGES:
        FACE_CAPTURE["done"] = True
        FACE_CAPTURE["active"] = False
        FACE_CAPTURE["message"] = "Face captured (not saved)"
        CAMERA_MODE = "attendance"


def camera_loop():
    global cap, camera_active, camera_error, latest_frame_jpeg

    try:
        cap_local = cv2.VideoCapture(camera_source)
        cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception as exc:
        cap_local = None
        camera_error = str(exc)

    if cap_local is None or not cap_local.isOpened():
        camera_active = False
        camera_error = "Unable to open camera."
        print(camera_error)
        return

    cap = cap_local
    camera_active = True
    camera_error = ""

    while True:
        ret, frame = cap_local.read()
        if not ret:
            camera_active = False
            camera_error = "Camera frame read failed."
            time.sleep(0.05)
            continue

        camera_active = True
        frame = cv2.flip(frame, 1)

        try:
            if CAMERA_MODE == "attendance":
                process_attendance_frame(frame)
            elif CAMERA_MODE == "admin_capture":
                process_capture_frame(frame)
        except Exception as exc:
            camera_error = str(exc)
            print("Camera processing error:", exc)

        ret2, jpeg = cv2.imencode(".jpg", frame)
        if ret2:
            with latest_frame_lock:
                latest_frame_jpeg = jpeg.tobytes()

        time.sleep(0.02)


@app.on_event("startup")
def startup_event():
    init_default_admin()
    load_runtime_settings()
    threading.Thread(target=camera_loop, daemon=True).start()


def mjpeg_generator():
    global latest_frame_jpeg

    while True:
        with latest_frame_lock:
            frame = latest_frame_jpeg
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, jpeg = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/last_event")
def api_last_event():
    return {"text": last_event_text, "ts": last_event_ts}


@app.get("/")
def root():
    return RedirectResponse(url="/attendance", status_code=302)


@app.get("/attendance", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "droid_default": CAM_DEFAULT_DROID_URL})


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    if not is_admin(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("admin.html", {"request": request, "logged_in": True})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_admin(request):
        return RedirectResponse(url="/admin", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    db = SessionLocal()
    try:
        settings = get_or_create_settings(db)
        stored_password = settings.admin_password
        if PasswordHasher.verify(password, stored_password):
            if PasswordHasher.needs_rehash(stored_password):
                settings.admin_password = PasswordHasher.hash(password)
                db.commit()
                load_runtime_settings()

            response = RedirectResponse(url="/admin", status_code=302)
            response.set_cookie(
                "admin_token",
                session_manager.create_token(),
                max_age=ADMIN_SESSION_SECONDS,
                httponly=True,
                samesite="lax",
            )
            record_audit("admin_login", "session", detail="Admin login successful")
            return response
    finally:
        db.close()

    record_audit("admin_login", "session", detail="Invalid password", success=False)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid password"},
        status_code=401,
    )


@app.get("/admin/logout")
def admin_logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("admin_token")
    return response


@app.get("/api/dashboard")
def dashboard(request: Request):
    require_admin(request)

    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.name.asc()).all()
        today = date_cls.today()
        summaries = [
            att_manager.summarize_day(
                name=user.name,
                day=today,
                events=load_user_day_events(db, user.id, today),
                user_id=user.id,
            )
            for user in users
        ]
        recent_rows = (
            db.query(AttendanceRow)
            .order_by(AttendanceRow.ts.desc())
            .limit(10)
            .all()
        )

        present = [item for item in summaries if item.events_count > 0]
        absent = [item for item in summaries if item.status == "absent"]
        late = [item for item in summaries if item.late]
        early = [item for item in summaries if item.early]

        return {
            "today": today.isoformat(),
            "metrics": {
                "total_users": len(users),
                "configured_faces": sum(1 for user in users if user.face_dir and os.path.exists(user.face_dir)),
                "present_today": len(present),
                "absent_today": len(absent),
                "late_today": len(late),
                "early_today": len(early),
            },
            "system": {
                "camera_active": camera_active,
                "camera_error": camera_error,
                "face_model_loaded": bool(face_model and face_model.has_model()),
                "liveness_model_loaded": bool(liveness_model and liveness_model.has_model()),
                "recognition_threshold": float(threshold),
            },
            "recent_events": [
                {
                    "user_id": row.user_id,
                    "name": row.name,
                    "action": row.action,
                    "ts": row.ts.isoformat() if row.ts else None,
                }
                for row in recent_rows
            ],
        }
    finally:
        db.close()


@app.get("/api/users")
def list_users(request: Request):
    require_admin(request)

    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.id.asc()).all()
        return [
            {
                "id": user.id,
                "name": user.name,
                "role": user.role,
                "configured": bool(user.face_dir and os.path.exists(user.face_dir)),
            }
            for user in users
        ]
    finally:
        db.close()


@app.put("/api/users/{user_id}")
def update_user(request: Request, user_id: int, name: str = Form(...), role: str = Form(...)):
    require_admin(request)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(404, "User not found")

        ensure_not_last_admin(db, user, role)
        user.name = name.strip()
        user.role = role
        db.commit()
        record_audit("update_user", "user", user_id, detail=f"name={user.name}, role={user.role}")
        return {"ok": True}
    finally:
        db.close()


@app.delete("/api/users/{user_id}")
def delete_user(request: Request, user_id: int):
    require_admin(request)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        ensure_not_last_admin(db, user, "deleted")
        db.query(AttendanceRow).filter(AttendanceRow.user_id == user.id).delete(synchronize_session=False)

        person_path = os.path.join(DATASET_DIR, str(user.id))
        if os.path.exists(person_path):
            shutil.rmtree(person_path, ignore_errors=True)

        db.delete(user)
        db.commit()
    finally:
        db.close()

    try:
        face_model.train()
        reload_face_runtime()
    except Exception as exc:
        record_audit("retrain_face_model", "model", detail=str(exc), success=False)

    record_audit("delete_user", "user", user_id)
    return {"ok": True}


@app.post("/api/capture/start")
def start_temp_capture(request: Request):
    require_admin(request)
    global CAMERA_MODE

    FACE_CAPTURE.clear()
    FACE_CAPTURE.update({
        "active": True,
        "frames": [],
        "count": 0,
        "progress": 0,
        "done": False,
        "error": False,
        "message": "Align your face inside the frame",
    })

    CAMERA_MODE = "admin_capture"
    return {"ok": True}


@app.post("/api/users/save")
def save_user(request: Request, name: str = Form(...), role: str = Form("employee")):
    require_admin(request)

    frames = FACE_CAPTURE.get("frames", [])
    if not frames:
        raise HTTPException(400, "No face data")

    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(400, "Name is required")

    db = SessionLocal()
    try:
        user = User(name=clean_name, role=role)
        db.add(user)
        db.commit()
        db.refresh(user)

        person_path = os.path.join(DATASET_DIR, str(user.id))
        os.makedirs(person_path, exist_ok=True)
        for index, image in enumerate(frames):
            cv2.imwrite(os.path.join(person_path, f"{user.id}_{index}.jpg"), image)

        user.face_dir = person_path
        db.commit()
        user_id = user.id
    finally:
        db.close()

    FACE_CAPTURE.clear()
    face_model.train()
    reload_face_runtime()
    record_audit("create_user", "user", user_id, detail=f"name={clean_name}, role={role}")
    return {"ok": True}


@app.get("/api/capture/status")
def capture_status(request: Request):
    require_admin(request)
    return {
        "progress": FACE_CAPTURE.get("progress", 0),
        "message": FACE_CAPTURE.get("message", ""),
        "done": FACE_CAPTURE.get("done", False),
        "error": FACE_CAPTURE.get("error", False),
    }


@app.post("/api/capture/cancel")
def cancel_capture(request: Request):
    require_admin(request)
    FACE_CAPTURE.clear()
    return {"ok": True}


@app.post("/api/users/{user_id}/capture/commit")
def commit_capture(request: Request, user_id: int):
    require_admin(request)

    frames = FACE_CAPTURE.get("frames", [])
    if not frames:
        raise HTTPException(400, "No captured face data")

    person_path = os.path.join(DATASET_DIR, str(user_id))
    os.makedirs(person_path, exist_ok=True)
    for index, image in enumerate(frames):
        cv2.imwrite(os.path.join(person_path, f"{user_id}_{index}.jpg"), image)

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.face_dir = person_path
            db.commit()
    finally:
        db.close()

    FACE_CAPTURE.clear()
    face_model.train()
    reload_face_runtime()
    record_audit("capture_user_face", "user", user_id)
    return {"ok": True}


@app.get("/api/attendance")
def get_attendance(
    request: Request,
    name: Optional[str] = None,
    id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    require_admin(request)

    db = SessionLocal()
    try:
        query = db.query(AttendanceRow)
        if name:
            query = query.filter(AttendanceRow.name.ilike(f"%{name}%"))
        if id:
            query = query.filter(AttendanceRow.user_id == id)
        if from_date:
            query = query.filter(AttendanceRow.ts >= datetime.strptime(from_date, "%Y-%m-%d"))
        if to_date:
            query = query.filter(AttendanceRow.ts < datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1))

        rows = query.order_by(AttendanceRow.ts.desc()).limit(1000).all()
        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "name": row.name,
                "action": row.action,
                "ts": row.ts.isoformat() if row.ts else None,
            }
            for row in rows
        ]
    finally:
        db.close()


@app.get("/api/attendance/history")
def attendance_history(
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    name: Optional[str] = None,
    id: Optional[int] = None,
    status: Optional[str] = None,
    late_early: Optional[str] = None,
):
    require_admin(request)

    today = date_cls.today()
    from_date = from_date or today.strftime("%Y-%m-%d")
    to_date = to_date or today.strftime("%Y-%m-%d")

    try:
        start_day = datetime.strptime(from_date, "%Y-%m-%d").date()
        end_day = datetime.strptime(to_date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    if end_day < start_day:
        raise HTTPException(status_code=400, detail="to_date must be after from_date")
    if (end_day - start_day).days > 31:
        raise HTTPException(status_code=400, detail="Maximum date range is 31 days")

    db = SessionLocal()
    try:
        user_query = db.query(User)
        if name:
            user_query = user_query.filter(User.name.ilike(f"%{name}%"))
        if id:
            user_query = user_query.filter(User.id == id)
        users = user_query.order_by(User.name.asc()).all()
        if not users:
            return []

        user_ids = [user.id for user in users]
        start_dt = datetime.combine(start_day, dtime.min)
        end_dt = datetime.combine(end_day + timedelta(days=1), dtime.min)

        rows = (
            db.query(AttendanceRow)
            .filter(
                AttendanceRow.ts >= start_dt,
                AttendanceRow.ts < end_dt,
                AttendanceRow.user_id.in_(user_ids),
            )
            .order_by(AttendanceRow.ts.asc())
            .all()
        )

        grouped: dict[tuple[date_cls, int], list[AttendanceEvent]] = {}
        for row in rows:
            if not row.ts or row.user_id is None:
                continue
            grouped.setdefault((row.ts.date(), row.user_id), []).append(AttendanceEvent.from_row(row))

        output = []
        for day in date_range(start_day, end_day):
            for user in users:
                summary = att_manager.summarize_day(
                    name=user.name,
                    day=day,
                    events=grouped.get((day, user.id), []),
                    user_id=user.id,
                )
                item = summary.to_dict()

                if status and status != "all" and item["status"] != status:
                    continue
                if late_early == "late" and not item["late"]:
                    continue
                if late_early == "early" and not item["early"]:
                    continue

                output.append(item)

        output.sort(key=lambda item: (item["date"], item.get("name") or ""), reverse=True)
        return output
    finally:
        db.close()


@app.get("/api/attendance/details")
def attendance_details(request: Request, user_id: int, date: str):
    require_admin(request)

    try:
        day = datetime.strptime(date, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")

    start_dt, end_dt = day_bounds(day)
    db = SessionLocal()
    try:
        rows = (
            db.query(AttendanceRow)
            .filter(
                AttendanceRow.user_id == user_id,
                AttendanceRow.ts >= start_dt,
                AttendanceRow.ts < end_dt,
            )
            .order_by(AttendanceRow.ts.asc())
            .all()
        )
        return [
            {
                "action": row.action,
                "ts": row.ts.isoformat() if row.ts else None,
                "raw_time": row.raw_time,
            }
            for row in rows
        ]
    finally:
        db.close()


@app.get("/api/settings")
def get_settings(request: Request):
    require_admin(request)

    db = SessionLocal()
    try:
        settings = get_or_create_settings(db)
        return {
            "shift_start": settings.shift_start,
            "shift_end": settings.shift_end,
            "min_in_to_out": settings.min_in_to_out,
            "min_out_to_in": settings.min_out_to_in,
            "overlay_duration": settings.overlay_duration,
            "camera_source": settings.camera_source,
            "admin_password": "",
            "admin_password_configured": bool(settings.admin_password),
        }
    finally:
        db.close()


@app.post("/api/settings")
def update_settings(
    request: Request,
    shift_start: str = Form(...),
    shift_end: str = Form(...),
    min_in_to_out: int = Form(...),
    min_out_to_in: int = Form(...),
    overlay_duration: int = Form(...),
    admin_password: str = Form(""),
    camera_source_form: str = Form("", alias="camera_source"),
):
    require_admin(request)

    parse_hhmm(shift_start)
    parse_hhmm(shift_end)
    if min_in_to_out < 0 or min_out_to_in < 0 or overlay_duration < 1:
        raise HTTPException(status_code=400, detail="Invalid numeric settings")

    db = SessionLocal()
    try:
        settings = get_or_create_settings(db)
        settings.shift_start = shift_start
        settings.shift_end = shift_end
        settings.min_in_to_out = min_in_to_out
        settings.min_out_to_in = min_out_to_in
        settings.overlay_duration = overlay_duration
        if camera_source_form:
            settings.camera_source = camera_source_form
        if admin_password.strip():
            settings.admin_password = PasswordHasher.hash(admin_password.strip())
        db.add(settings)
        db.commit()
    finally:
        db.close()

    load_runtime_settings()
    record_audit("update_settings", "settings", detail="Runtime settings updated")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    attendance_url = "http://127.0.0.1:8000/attendance"
    print(f"Attendance page: {attendance_url}")
    threading.Timer(1.5, webbrowser.open, args=(attendance_url,)).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
