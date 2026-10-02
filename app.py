import os
import io
import time
import threading
import subprocess
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np
import imageio_ffmpeg

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# BASIC CONFIG
# =========================================================

APP_NAME = "CCTV Water Monitoring System"

THAILAND_TZ = timezone(timedelta(hours=7))

app = FastAPI(
    title=APP_NAME,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CAMERA CONFIG
# =========================================================

CAMERAS = {
    "CCTV-01": {
        "id": "CCTV-01",
        "enabled": True,

        "name": "เทศบาลนครนนทบุรี (เมือง)",
        "location": "ท่าน้ำนนท์",
        "title": "เทศบาลนครนนทบุรี (เมือง)",
        "description": "จุดวัด ท่าน้ำนนท์",

        "stream": "https://stream.firsttech.co.th/live/nakornnont.stream/playlist.m3u8",

        # โลโก้หลักของจังหวัด
        "provinceLogo": "https://upload.wikimedia.org/wikipedia/commons/1/15/Seal_Nonthaburi.png",

        # โลโก้เทศบาล
        "municipalityLogo": "https://nakornnont.go.th/images/content/logo-139-1/logo.png",

        # โลโก้หน่วยงานผู้ดูแลระบบ
        "agencyLogo": "https://cctv-nont.firsttech.co.th/img/FirstTech_Logo.e48d7620.png",

        "agencyName": "Firsttech Design Co., Ltd.",

        # รูปสำหรับกล่องข้อมูลระดับน้ำ
        # ใส่ URL รูปของคุณเองภายหลังได้
        "waterStatusImage": "",

        # ข้อมูลระดับน้ำแบบ MANUAL
        "water": {
            "mode": "manual",

            # ถ้าไม่มีข้อมูล ให้ใส่ None
            "value": 32,

            "unit": "ซม.",

            "status": "critical",
            "statusText": "วิกฤต",

            "updatedAt": 2026-10-02T10:07:00+07:00,

            "message": "บันทึกข้อมูลระดับน้ำ"
        }
    },

    "CCTV-02": {
        "id": "CCTV-02",

        # ปิดการใช้งานชั่วคราว
        "enabled": False,

        "maintenance": True,

        "name": "เทศบาลนครปากเกร็ด",
        "location": "ท่าน้ำปากเกร็ด-หัวถนน",
        "title": "เทศบาลนครปากเกร็ด",
        "description": "จุดวัด ท่าน้ำปากเกร็ด-หัวถนน",

        "stream": "https://thaiclouderp.com/video/pakkret-river.m3u8",

        "provinceLogo": "https://upload.wikimedia.org/wikipedia/commons/1/15/Seal_Nonthaburi.png",

        "municipalityLogo": "https://thaiclouderp.com/video/asset/images/pakkret_logo.png",

        "agencyLogo": "https://thaiclouderp.com/video/asset/images/ccs_logo.png",

        "agencyName": "Cloud Computing Solutions Co., Ltd.",

        "waterStatusImage": "",

        "water": {
            "mode": "manual",
            "value": None,
            "unit": "ซม.",
            "status": "waiting",
            "statusText": "รอข้อมูล",
            "updatedAt": None,
            "message": "ระบบอยู่ระหว่างการปรับปรุง"
        }
    }
}


# =========================================================
# RUNTIME CACHE
# =========================================================

LATEST = {}

LOCK = threading.Lock()


# =========================================================
# TIME
# =========================================================

def now_thailand():
    return datetime.now(THAILAND_TZ)


def iso_now():
    return now_thailand().isoformat()


# =========================================================
# FFMPEG
# =========================================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def extract_frame(stream_url: str) -> bytes:
    """
    ดึงภาพ frame ปัจจุบันจาก HLS

    ไม่ต้องให้ browser เข้าไปอ่าน HLS โดยตรง
    server เป็นคนดึงภาพแทน
    """

    command = [
        FFMPEG,

        "-hide_banner",
        "-loglevel", "error",

        # จำกัดเวลารอ
        "-rw_timeout", "10000000",

        # HLS
        "-i", stream_url,

        # เอาแค่ 1 frame
        "-frames:v", "1",

        # ลดขนาดเพื่อประหยัด CPU
        "-vf", "scale=1280:-2",

        # ส่ง PNG ออกทาง stdout
        "-f", "image2pipe",
        "-vcodec", "png",

        "pipe:1"
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout while reading HLS stream")

    if result.returncode != 0 or not result.stdout:
        error = result.stderr.decode(
            "utf-8",
            errors="ignore"
        )

        raise RuntimeError(
            "FFmpeg could not read stream: " + error[-1000:]
        )

    return result.stdout


# =========================================================
# IMAGE DECODER
# =========================================================

def decode_image(image_bytes: bytes):
    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(
            "Cannot decode snapshot"
        )

    return image


# =========================================================
# WATER LEVEL DETECTOR
# =========================================================

def detect_water_level(image, config):
    """
    ระบบอ่านระดับน้ำแบบ Computer Vision

    IMPORTANT:
    ยังไม่เปิดใช้งานจนกว่าจะ calibrate ภาพจริง
    """

    if not config.get("enabled", False):
        return {
            "value": None,
            "unit": "cm",
            "status": "unavailable",
            "confidence": 0,
            "message": "ยังไม่ได้ตั้งค่าพื้นที่ตรวจวัด"
        }

    h, w = image.shape[:2]

    roi_config = config["roi"]

    x1 = int(w * roi_config["x"])
    y1 = int(h * roi_config["y"])

    x2 = int(
        w * (
            roi_config["x"] +
            roi_config["width"]
        )
    )

    y2 = int(
        h * (
            roi_config["y"] +
            roi_config["height"]
        )
    )

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "value": None,
            "unit": "cm",
            "status": "error",
            "confidence": 0,
            "message": "ROI ไม่ถูกต้อง"
        }

    # ---------------------------------------------
    # Blur ลด noise
    # ---------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ---------------------------------------------
    # Vertical gradient
    #
    # ผิวน้ำมักสร้าง transition ในภาพ
    # ---------------------------------------------

    gradient = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    strength = np.mean(
        np.abs(gradient),
        axis=1
    )

    # smoothing
    smooth = int(
        config.get("smooth", 9)
    )

    if smooth % 2 == 0:
        smooth += 1

    if smooth < 3:
        smooth = 3

    kernel = np.ones(
        smooth,
        dtype=np.float32
    ) / smooth

    strength = np.convolve(
        strength,
        kernel,
        mode="same"
    )

    # ไม่ใช้ขอบบน/ล่างสุด
    margin = max(
        5,
        int(len(strength) * 0.05)
    )

    search = strength[
        margin:
        len(strength) - margin
    ]

    if len(search) == 0:
        return {
            "value": None,
            "unit": "cm",
            "status": "unavailable",
            "confidence": 0,
            "message": "ไม่พบพื้นที่ตรวจสอบ"
        }

    local_index = int(
        np.argmax(search)
    )

    detected_y = (
        margin +
        local_index
    )

    # ---------------------------------------------
    # Convert pixel position
    # ---------------------------------------------

    roi_h = roi.shape[0]

    relative_y = (
        detected_y /
        max(1, roi_h - 1)
    )

    top_pixel = float(
        config.get("topPixel", 0.10)
    )

    bottom_pixel = float(
        config.get("bottomPixel", 0.90)
    )

    top_cm = float(
        config.get("topCm", 100)
    )

    bottom_cm = float(
        config.get("bottomCm", 0)
    )

    denominator = (
        bottom_pixel -
        top_pixel
    )

    if abs(denominator) < 0.0001:
        return {
            "value": None,
            "unit": "cm",
            "status": "error",
            "confidence": 0,
            "message": "Calibration ไม่ถูกต้อง"
        }

    ratio = (
        relative_y - top_pixel
    ) / denominator

    level = (
        top_cm +
        ratio * (
            bottom_cm -
            top_cm
        )
    )

    level = round(
        float(level),
        1
    )

    # ---------------------------------------------
    # confidence
    # ---------------------------------------------

    peak = float(
        np.max(search)
    )

    average = float(
        np.mean(search)
    )

    if average <= 0:
        confidence = 0
    else:
        confidence = min(
            100,
            max(
                0,
                int(
                    (
                        peak /
                        max(average, 0.001)
                    ) * 20
                )
            )
        )

    # ไม่รับค่าที่ confidence ต่ำมาก
    if confidence < 25:
        return {
            "value": None,
            "unit": "cm",
            "status": "uncertain",
            "confidence": confidence,
            "message": "ภาพยังไม่ชัดพอสำหรับยืนยันระดับน้ำ"
        }

    # ---------------------------------------------
    # Status
    #
    # สามารถเปลี่ยน threshold ได้ภายหลัง
    # ---------------------------------------------

    if level >= 70:
        status = "critical"
        status_text = "วิกฤต"

    elif level >= 50:
        status = "warning"
        status_text = "เฝ้าระวัง"

    else:
        status = "normal"
        status_text = "ปกติ"

    return {
        "value": level,
        "unit": "cm",
        "status": status,
        "statusText": status_text,
        "confidence": confidence,
        "message": "ตรวจพบระดับน้ำจากภาพ"
    }


# =========================================================
# CAMERA SNAPSHOT
# =========================================================

def get_camera_snapshot(camera_id: str):
    if camera_id not in CAMERAS:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบกล้อง"
        )

    camera = CAMERAS[camera_id]

    raw = extract_frame(
        camera["stream"]
    )

    image = decode_image(raw)

    water = detect_water_level(
        image,
        camera["water"]
    )

    timestamp = iso_now()

    result = {
        "camera": camera_id,

        "timestamp": timestamp,

        "water": water,

        "width": int(image.shape[1]),
        "height": int(image.shape[0])
    }

    with LOCK:
        LATEST[camera_id] = result

    return raw, result


# =========================================================
# HOME
# =========================================================

@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "time": iso_now()
    }


# =========================================================
# CAMERA LIST
# =========================================================

@app.get("/api/cameras")
def get_cameras():
    result = []

    for camera_id, camera in CAMERAS.items():
        result.append({
            "id": camera["id"],
            "enabled": camera.get("enabled", True),
            "maintenance": camera.get("maintenance", False),

            "name": camera.get("name", ""),
            "location": camera.get("location", ""),
            "title": camera.get("title", ""),
            "description": camera.get("description", ""),

            "stream": camera.get("stream", ""),

            "provinceLogo": camera.get("provinceLogo", ""),
            "municipalityLogo": camera.get("municipalityLogo", ""),
            "agencyLogo": camera.get("agencyLogo", ""),
            "agencyName": camera.get("agencyName", ""),

            "waterStatusImage": camera.get("waterStatusImage", ""),

            "water": camera.get("water", {})
        })

    return result


# =========================================================
# CURRENT STATUS
# =========================================================

@app.get("/api/status/{camera_id}")
def status(camera_id: str):

    if camera_id not in CAMERAS:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบกล้อง"
        )

    with LOCK:
        cached = LATEST.get(camera_id)

    if cached:
        return cached

    return {
        "camera": camera_id,
        "timestamp": None,
        "water": {
            "value": None,
            "unit": "cm",
            "status": "waiting",
            "confidence": 0,
            "message": "รอภาพจากกล้อง"
        }
    }


# =========================================================
# TAKE SNAPSHOT + ANALYZE
# =========================================================

@app.get("/api/snapshot/{camera_id}")
def snapshot(camera_id: str):

    raw, result = get_camera_snapshot(
        camera_id
    )

    return StreamingResponse(
        io.BytesIO(raw),
        media_type="image/png",
        headers={
            "X-Camera-ID": camera_id,
            "X-Snapshot-Time": result["timestamp"]
        }
    )


# =========================================================
# ANALYZE WITHOUT RETURNING IMAGE
# =========================================================

@app.get("/api/analyze/{camera_id}")
def analyze(camera_id: str):

    try:
        raw, result = get_camera_snapshot(
            camera_id
        )

        return JSONResponse(
            content=result
        )

    except Exception as error:

        return JSONResponse(
            status_code=503,
            content={
                "camera": camera_id,
                "timestamp": None,

                "water": {
                    "value": None,
                    "unit": "cm",
                    "status": "error",
                    "confidence": 0,
                    "message": str(error)
                }
            }
        )


# =========================================================
# SNAPSHOT IMAGE FROM CACHE
# =========================================================

@app.get("/api/snapshot-cached/{camera_id}")
def cached_snapshot(camera_id: str):

    if camera_id not in CAMERAS:
        raise HTTPException(
            status_code=404,
            detail="ไม่พบกล้อง"
        )

    try:

        raw = extract_frame(
            CAMERAS[camera_id]["stream"]
        )

        return StreamingResponse(
            io.BytesIO(raw),
            media_type="image/png"
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


# =========================================================
# STATIC WEBSITE
# =========================================================

app.mount(
    "/",
    StaticFiles(
        directory="static",
        html=True
    ),
    name="static"
)
