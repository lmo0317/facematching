"""
Shared helpers used by app.py, celebrity_service.py and populate_celebrities.py:
image normalization, LLM JSON extraction, ArcFace face embeddings and similarity-to-percent calibration.
"""

import os
import io
import re
import json
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("facematch.utils")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
ARCFACE_PATH = os.path.join(MODELS_DIR, "arcface_w600k_r50.onnx")
EMBEDDING_DIM = 512

HTTP_HEADERS = {
    "User-Agent": "FaceMatchApp/2.0 (https://minohlee.mooo.com; admin@minohlee.mooo.com)"
}

# ArcFace canonical 5-point template for a 112x112 crop (eyes, nose tip, mouth corners; image-left first).
# YuNet emits landmarks in the same order.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)
DETECT_MAX_DIM = 1280

_ARCFACE_SESSION = None


def image_bytes_to_data_url(image_bytes: bytes, max_dim: int = 768) -> str:
    """Apply EXIF orientation, convert to RGB, downscale to max_dim and return a JPEG data URL."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=88)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Strip thought tags from LLM output and parse the JSON object, or return None."""
    cleaned = strip_thought_tags(text)

    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except Exception:
            pass

    outer = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if outer:
        try:
            return json.loads(outer.group(1))
        except Exception:
            pass

    return None


def strip_thought_tags(text: str) -> str:
    cleaned = re.sub(r"<channel>thought.*?</channel>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|think\|>.*?</turn>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned


def decode_image_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode bytes to a BGR array with EXIF orientation applied (phone photos are often rotated)."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Failed to decode image: {e}")
        return None


def detect_faces_yunet(img_bgr: np.ndarray, score_threshold: float = 0.6) -> List[np.ndarray]:
    """YuNet face rows [x, y, w, h, 10 landmark coords, score] in original image coordinates."""
    if not os.path.exists(YUNET_PATH):
        return []
    h, w = img_bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_DIM / max(h, w))
    work = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
    wh, ww = work.shape[:2]
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (ww, wh), score_threshold, 0.3, 50)
    _, faces = detector.detect(work)
    if faces is None:
        return []
    rows = []
    for f in faces:
        f = f.copy()
        f[:14] /= scale
        rows.append(f)
    return rows


def compute_square_face_box(fx: int, fy: int, fw: int, fh: int, img_w: int, img_h: int, landmarks: Optional[List[float]] = None) -> Tuple[int, int, int, int]:
    """Calculate balanced, square (1:1) bounding box centered on head with natural hair and chin margins."""
    face_dim = max(fw, fh)
    # Headshot crop: comfortable margin for hair headroom, ears, and chin/collar
    crop_size = int(face_dim * 1.65)
    crop_size = min(crop_size, img_w, img_h)
    crop_size = max(crop_size, 30)

    if landmarks is not None and len(landmarks) >= 4:
        # Landmarks: right eye (rx, ry), left eye (lx, ly)
        rx, ry, lx, ly = landmarks[0:4]
        eye_cx = (rx + lx) / 2.0
        eye_cy = (ry + ly) / 2.0
        cx = int(eye_cx)
        # Position eyes at ~38% from the top of the crop for balanced portrait framing
        py = int(eye_cy - int(crop_size * 0.38))
        px = int(cx - crop_size // 2)
    else:
        cx = fx + fw // 2
        cy = fy + int(fh * 0.45)
        px = cx - crop_size // 2
        py = cy - int(crop_size * 0.45)

    # Strictly clamp within image boundary while preserving 1:1 square ratio
    px = max(0, min(img_w - crop_size, px))
    py = max(0, min(img_h - crop_size, py))

    return px, py, crop_size, crop_size


def get_arcface_session():
    global _ARCFACE_SESSION
    if _ARCFACE_SESSION is None and os.path.exists(ARCFACE_PATH):
        try:
            import onnxruntime as ort
            _ARCFACE_SESSION = ort.InferenceSession(ARCFACE_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            logger.warning(f"Failed to load ArcFace model: {e}")
    return _ARCFACE_SESSION


def align_face(img_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """Similarity-warp the face to the 112x112 ArcFace template using its 5 landmarks."""
    src = np.asarray(face_row[4:14], dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img_bgr, matrix, (112, 112), borderValue=0.0)


def embed_face(img_bgr: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
    """512-D L2-normalized ArcFace embedding of one detected face."""
    session = get_arcface_session()
    if session is None:
        return None
    aligned = align_face(img_bgr, face_row)
    blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    feat = session.run(None, {session.get_inputs()[0].name: blob})[0].flatten()
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else None


def primary_face(img_bgr: np.ndarray, score_threshold: float = 0.6) -> Optional[np.ndarray]:
    faces = detect_faces_yunet(img_bgr, score_threshold)
    return max(faces, key=lambda f: f[2] * f[3]) if faces else None


def extract_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """ArcFace embedding of the largest face in the image, or None if no face/model."""
    try:
        img_bgr = decode_image_bgr(image_bytes)
        if img_bgr is None:
            return None
        face = primary_face(img_bgr)
        return embed_face(img_bgr, face) if face is not None else None
    except Exception as e:
        logger.warning(f"Error extracting face embedding: {e}")
        return None


# ArcFace cosine -> human-facing percentage, piecewise-linear between anchors.
# Anchors come from tools/calibrate_scores.py (2026-09 run, single photo vs single photo):
#   strangers (3000 pairs)   p50 0.008  p90 0.091  p95 0.114
#   kin (740 real pairs)     p25 0.076  p50 0.136  p75 0.204  p90 0.285  p95 0.351
#   same person (567 pairs)  p5 0.325   p10 0.389  p25 0.473  p50 0.567
#   celebrity top-1 vs DB    p5 0.243   p50 0.327  p90 0.393  p95 0.414
SIMILARITY_ANCHORS = {
    # Two photos, "how alike do these two people look". Verdicts: >=82 판박이, >=68 매우 높음, >=50 은근한, >=35 낮음.
    # stranger median -> ~20 (남남), kin median -> ~54 (은근한), kin p75 -> ~68, kin p90 -> ~81, same person -> 95+
    "family": [(-1.0, 3), (-0.05, 10), (0.0, 18), (0.05, 30), (0.10, 42), (0.14, 55), (0.20, 68),
               (0.28, 80), (0.35, 87), (0.45, 92), (0.60, 96), (1.0, 99)],
    # Two photos, "is this the same person". Verdicts: >=85 확실, >=70 유력, >=50 가능성 낮음.
    # kin p95 -> ~63 (not "유력"), same-person p10 -> ~71, same-person p25 -> ~85
    "identical": [(-1.0, 1), (0.0, 3), (0.15, 15), (0.25, 35), (0.32, 55), (0.38, 70), (0.47, 85),
                  (0.60, 94), (0.80, 98), (1.0, 99)],
    # One photo vs a celebrity's multi-photo mean. Typical best match (p50) -> ~72, top 5% -> ~84, same person -> 95+
    "celebrity": [(-1.0, 5), (0.0, 20), (0.15, 45), (0.24, 60), (0.30, 68), (0.33, 72), (0.37, 78),
                  (0.42, 85), (0.50, 91), (0.60, 95), (1.0, 99)],
}


def similarity_to_percent(cosine: float, profile: str = "family") -> int:
    xs, ys = zip(*SIMILARITY_ANCHORS.get(profile, SIMILARITY_ANCHORS["family"]))
    return int(round(float(np.interp(cosine, xs, ys))))


def adjust_part_scores(parts: Optional[Dict[str, Any]], overall: int, keys: List[str]) -> Dict[str, int]:
    """
    Re-center LLM per-part scores on the measured overall score: keep which parts the LLM thought
    were more/less alike, but not its absolute level (which is poorly calibrated).
    """
    values = {}
    for k in keys:
        try:
            values[k] = float((parts or {}).get(k))
        except (TypeError, ValueError):
            pass
    if len(values) < len(keys):
        return {k: overall for k in keys}
    center = sum(values.values()) / len(values)
    return {k: int(max(5, min(99, round(overall + (values[k] - center) * 0.5)))) for k in keys}


def data_url_to_bytes(data: str) -> bytes:
    return base64.b64decode(data.split(",", 1)[1] if "," in data else data)
