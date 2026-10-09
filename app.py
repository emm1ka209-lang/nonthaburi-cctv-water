"""CCTV Water Monitoring System - Nonthaburi (backend)

ค่าตั้งค่าผ่าน environment variables:
  ADMIN_TOKEN      รหัสสำหรับอัปเดตระดับน้ำผ่าน API (ไม่ตั้ง = ปิดการแก้ไข)
  ALLOWED_ORIGINS  โดเมนที่อนุญาตให้เรียก API ข้ามโดเมน คั่นด้วย , (ค่าเริ่มต้น: เฉพาะโดเมนเดียวกัน)
  SNAPSHOT_TTL     อายุแคชภาพ snapshot เป็นวินาที (ค่าเริ่มต้น 10)
  DATA_DIR         โฟลเดอร์เก็บ water.json (ค่าเริ่มต้น ./data)
"""
import hmac
import io
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_NAME = "CCTV Water Monitoring System"
TZ = timezone(timedelta(hours=7))
BASE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE / "data"))
WATER_FILE = DATA_DIR / "water.json"
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
SNAPSHOT_TTL = float(os.getenv("SNAPSHOT_TTL", "10"))
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

log = logging.getLogger("nontwater")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title=APP_NAME, version="2.0.0")
app.add_middleware(GZipMiddleware, minimum_size=800)
if ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

# ---------------------------------------------------------------- cameras
# ข้อมูลกล้อง (ไม่เปลี่ยนบ่อย) อยู่ที่นี่ / ระดับน้ำย้ายไป data/water.json
CAMERAS = {
    "CCTV-01": {
        "id": "CCTV-01",
        "enabled": True,
        "name": "เทศบาลนครนนทบุรี (เมือง)",
        "location": "ท่าน้ำนนท์",
        "stream": "https://stream.firsttech.co.th/live/nakornnont.stream/index.m3u8?cookieCheck=1",
        "thumbnail": "/img/cctv01.jpg",
        "provinceLogo": "https://upload.wikimedia.org/wikipedia/commons/1/15/Seal_Nonthaburi.png",
        "municipalityLogo": "https://nakornnont.go.th/images/content/logo-139-1/logo.png",
        "agencyLogo": "https://cctv-nont.firsttech.co.th/img/FirstTech_Logo.e48d7620.png",
        "agencyName": "Firsttech Design Co., Ltd.",
        # เปิดใช้เมื่อ calibrate ภาพจริงแล้วเท่านั้น
        "vision": {"enabled": False},
    },
    "CCTV-02": {
        "id": "CCTV-02",
        "enabled": True,
        "name": "เทศบาลนครปากเกร็ด (ปากเกร็ด)",
        "location": "ท่าน้ำปากเกร็ด-หัวถนน",
        "stream": "https://thaiclouderp.com/video/pakkret-river.m3u8",
        "thumbnail": "/img/cctv02.jpg",
        "provinceLogo": "https://upload.wikimedia.org/wikipedia/commons/1/15/Seal_Nonthaburi.png",
        "municipalityLogo": "https://thaiclouderp.com/video/asset/images/pakkret_logo.png",
        "agencyLogo": "https://thaiclouderp.com/video/asset/images/ccs_logo.png",
        "agencyName": "Cloud Computing Solutions Co., Ltd.",
        "vision": {"enabled": False},
    },
}

# ธงเตือนระดับน้ำ: ใช้ที่เดียว ให้ status กับข้อความสอดคล้องกันเสมอ
FLAGS = {
    "normal": "ปกติ ธงเขียว",
    "warning": "เฝ้าระวัง ธงเหลือง",
    "critical": "วิกฤต ธงแดง",
}

DEFAULT_WATER = {
    "CCTV-01": {"value": 36.5, "status": "critical", "updatedAt": "2026-10-02T20:31:01+07:00"},
    "CCTV-02": {"value": 27.5, "status": "warning", "updatedAt": "2026-10-02T20:31:01+07:00"},
}

# ---------------------------------------------------------------- water store
_water_lock = threading.Lock()


def _build_water(raw: dict) -> dict:
    status = raw.get("status", "normal")
    return {
        "mode": "manual",
        "value": raw.get("value"),
        "unit": "ซม.",
        "status": status,
        "statusText": FLAGS.get(status, "ไม่ทราบสถานะ"),
        "updatedAt": raw.get("updatedAt"),
        "message": raw.get("message", "บันทึกข้อมูลระดับน้ำ"),
    }


def _load_water() -> dict:
    try:
        return json.loads(WATER_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(DEFAULT_WATER)
    except Exception:
        log.exception("water.json อ่านไม่ได้ ใช้ค่าเริ่มต้นแทน")
        return dict(DEFAULT_WATER)


def _save_water(data: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WATER_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, WATER_FILE)  # เขียนแบบ atomic ไฟล์ไม่พังถ้าดับกลางทาง


def get_camera(camera_id: str) -> dict:
    cam = CAMERAS.get(camera_id)
    if not cam:
        raise HTTPException(404, "ไม่พบกล้อง")
    return cam


# ---------------------------------------------------------------- snapshot
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_snap_cache: dict[str, tuple[float, bytes]] = {}
_snap_locks = {cid: threading.Lock() for cid in CAMERAS}


def extract_frame(url: str) -> bytes:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-rw_timeout", "10000000", "-i", url,
        "-frames:v", "1", "-an", "-vf", "scale=1280:-2",
        "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "4", "pipe:1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        raise RuntimeError("อ่านสตรีมหมดเวลา")
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError("อ่านสตรีมไม่ได้: " + r.stderr.decode("utf-8", "ignore")[-300:])
    return r.stdout


def cached_frame(camera_id: str) -> bytes:
    """คืนภาพล่าสุดจากแคช; ถ้าหมดอายุจะดึงใหม่ครั้งเดียว (คำขอซ้อนกันรอผลเดียวกัน)
    ป้องกันการเปิด ffmpeg ถี่ ๆ ซึ่งกิน CPU บน free plan"""
    cam = get_camera(camera_id)
    hit = _snap_cache.get(camera_id)
    if hit and time.time() - hit[0] < SNAPSHOT_TTL:
        return hit[1]
    with _snap_locks[camera_id]:
        hit = _snap_cache.get(camera_id)
        if hit and time.time() - hit[0] < SNAPSHOT_TTL:
            return hit[1]
        try:
            frame = extract_frame(cam["stream"])
        except RuntimeError:
            if hit:  # สตรีมสะดุด: ให้ภาพเก่าไปก่อนดีกว่า error
                return hit[1]
            raise
        _snap_cache[camera_id] = (time.time(), frame)
        return frame


# ---------------------------------------------------------------- vision (ยังปิดอยู่)
def detect_water_level(image, cfg: dict) -> dict:
    """อ่านระดับน้ำจากภาพ: หาแนวขอบน้ำใน ROI ด้วย vertical gradient
    ต้องตั้ง vision = {enabled, roi:{x,y,width,height}, topPixel, bottomPixel, topCm, bottomCm}"""
    if not cfg.get("enabled"):
        return {"value": None, "status": "unavailable", "confidence": 0,
                "message": "ยังไม่ได้ตั้งค่าพื้นที่ตรวจวัด"}
    h, w = image.shape[:2]
    r = cfg["roi"]
    x1, y1 = int(w * r["x"]), int(h * r["y"])
    x2, y2 = int(w * (r["x"] + r["width"])), int(h * (r["y"] + r["height"]))
    roi = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if roi.size == 0:
        return {"value": None, "status": "error", "confidence": 0, "message": "ROI ไม่ถูกต้อง"}

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    strength = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)).mean(axis=1)
    k = max(3, int(cfg.get("smooth", 9)) | 1)
    strength = np.convolve(strength, np.ones(k) / k, mode="same")
    margin = max(5, int(len(strength) * 0.05))
    search = strength[margin:len(strength) - margin]
    if len(search) == 0:
        return {"value": None, "status": "unavailable", "confidence": 0, "message": "ไม่พบพื้นที่ตรวจสอบ"}

    rel_y = (margin + int(np.argmax(search))) / max(1, roi.shape[0] - 1)
    top_px, bot_px = cfg.get("topPixel", 0.10), cfg.get("bottomPixel", 0.90)
    top_cm, bot_cm = cfg.get("topCm", 100), cfg.get("bottomCm", 0)
    if abs(bot_px - top_px) < 1e-4:
        return {"value": None, "status": "error", "confidence": 0, "message": "Calibration ไม่ถูกต้อง"}
    level = round(float(top_cm + (rel_y - top_px) / (bot_px - top_px) * (bot_cm - top_cm)), 1)

    avg = float(search.mean())
    confidence = 0 if avg <= 0 else min(100, int(float(search.max()) / avg * 20))
    if confidence < 25:
        return {"value": None, "status": "uncertain", "confidence": confidence,
                "message": "ภาพยังไม่ชัดพอสำหรับยืนยันระดับน้ำ"}
    status = "critical" if level >= 70 else "warning" if level >= 50 else "normal"
    return {"value": level, "unit": "cm", "status": status, "statusText": FLAGS[status],
            "confidence": confidence, "message": "ตรวจพบระดับน้ำจากภาพ"}


# ---------------------------------------------------------------- API
def public_camera(cam: dict, water: dict) -> dict:
    out = {k: v for k, v in cam.items() if k != "vision"}
    out["water"] = _build_water(water.get(cam["id"], {}))
    return out


@app.get("/api/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": datetime.now(TZ).isoformat()}


@app.get("/api/cameras")
def list_cameras(response: Response):
    response.headers["Cache-Control"] = "no-store"
    with _water_lock:
        water = _load_water()
    return [public_camera(c, water) for c in CAMERAS.values()]


@app.get("/api/snapshot/{camera_id}")
def snapshot(camera_id: str):
    try:
        frame = cached_frame(camera_id)
    except RuntimeError as e:
        log.warning("snapshot %s ล้มเหลว: %s", camera_id, e)
        raise HTTPException(503, "ดึงภาพจากกล้องไม่สำเร็จ")
    return Response(frame, media_type="image/jpeg",
                    headers={"Cache-Control": f"public, max-age={int(SNAPSHOT_TTL)}"})


@app.get("/api/analyze/{camera_id}")
def analyze(camera_id: str):
    cam = get_camera(camera_id)
    try:
        frame = cached_frame(camera_id)
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("ถอดรหัสภาพไม่ได้")
        result = detect_water_level(image, cam.get("vision", {}))
        return {"camera": camera_id, "timestamp": datetime.now(TZ).isoformat(), "water": result,
                "width": int(image.shape[1]), "height": int(image.shape[0])}
    except RuntimeError as e:
        return JSONResponse(status_code=503, content={
            "camera": camera_id, "timestamp": None,
            "water": {"value": None, "status": "error", "confidence": 0, "message": str(e)}})


class WaterUpdate(BaseModel):
    value: float = Field(ge=0, le=1000, description="ระดับน้ำ (ซม.)")
    status: str = Field(pattern="^(normal|warning|critical)$")
    message: str | None = Field(default=None, max_length=200)


def require_admin(authorization: str = Header(default="")):
    if not ADMIN_TOKEN:
        raise HTTPException(503, "ยังไม่ได้ตั้งค่า ADMIN_TOKEN บนเซิร์ฟเวอร์")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "รหัสไม่ถูกต้อง")


@app.post("/api/water/{camera_id}", dependencies=[Depends(require_admin)])
def update_water(camera_id: str, body: WaterUpdate):
    get_camera(camera_id)
    entry = {"value": body.value, "status": body.status,
             "updatedAt": datetime.now(TZ).isoformat(timespec="seconds")}
    if body.message:
        entry["message"] = body.message
    with _water_lock:
        data = _load_water()
        data[camera_id] = entry
        _save_water(data)
    log.info("อัปเดตระดับน้ำ %s = %s ซม. (%s)", camera_id, body.value, body.status)
    return _build_water(entry)


# ---------------------------------------------------------------- static
app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
