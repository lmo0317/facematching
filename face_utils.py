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


# Head-and-shoulders framing: celebrity display photos and the CLIP visual index are built with it
HEAD_CROP_SCALE, HEAD_CROP_EYE = 1.65, 0.38
# Face-focused framing for the UI cropper and extra-photo auto crops (users found 1.65 too loose).
# Face re-detection and ArcFace embeddings are unchanged down to ~1.2x (tested on 250 photos).
FACE_CROP_SCALE, FACE_CROP_EYE = 1.3, 0.42


def compute_square_face_box(fx: int, fy: int, fw: int, fh: int, img_w: int, img_h: int,
                            landmarks: Optional[List[float]] = None, scale: float = HEAD_CROP_SCALE,
                            eye_ratio: float = HEAD_CROP_EYE) -> Tuple[int, int, int, int]:
    """Square (1:1) box of `scale` x face size, eyes placed `eye_ratio` from the top, clamped to the image."""
    face_dim = max(fw, fh)
    crop_size = int(face_dim * scale)
    crop_size = min(crop_size, img_w, img_h)
    crop_size = max(crop_size, 30)

    if landmarks is not None and len(landmarks) >= 4:
        # Landmarks: right eye (rx, ry), left eye (lx, ly)
        rx, ry, lx, ly = landmarks[0:4]
        eye_cx = (rx + lx) / 2.0
        eye_cy = (ry + ly) / 2.0
        cx = int(eye_cx)
        py = int(eye_cy - int(crop_size * eye_ratio))
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
    feats = extract_face_features(image_bytes, with_visual=False)
    return feats[0] if feats else None


def extract_face_features(image_bytes: bytes, with_visual: bool = True) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """(ArcFace identity embedding, CLIP head-crop embedding or None) of the largest face, or None."""
    try:
        img_bgr = decode_image_bgr(image_bytes)
        if img_bgr is None:
            return None
        face = primary_face(img_bgr)
        if face is None:
            return None
        identity = embed_face(img_bgr, face)
        if identity is None:
            return None
        return identity, (visual_embedding(img_bgr, face) if with_visual else None)
    except Exception as e:
        logger.warning(f"Error extracting face features: {e}")
        return None


def best_matching_embedding(image_bytes: bytes, reference: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """(embedding, cosine) of the face in the image most similar to `reference` (e.g. a group photo), or None."""
    try:
        img_bgr = decode_image_bgr(image_bytes)
        if img_bgr is None:
            return None
        best = None
        for row in detect_faces_yunet(img_bgr, 0.6):
            emb = embed_face(img_bgr, row)
            if emb is not None and (best is None or float(emb @ reference) > best[1]):
                best = (emb, float(emb @ reference))
        return best
    except Exception as e:
        logger.warning(f"Error matching faces: {e}")
        return None


# ---- Visual impression (CLIP) -------------------------------------------------------------
# ArcFace deliberately ignores hair, makeup, expression and pose, but people judge "looks alike"
# from exactly those. CLIP on the same head crop the UI shows is used to order near-tied matches.
CLIP_PATH = os.path.join(MODELS_DIR, "clip_vit_b32_vision_q.onnx")
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CLIP_SESSION = None


def get_clip_session():
    global _CLIP_SESSION
    if _CLIP_SESSION is None and os.path.exists(CLIP_PATH):
        try:
            import onnxruntime as ort
            _CLIP_SESSION = ort.InferenceSession(CLIP_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
    return _CLIP_SESSION


def head_crop(img_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """Square face+hair crop, same framing as the celebrity display photos."""
    h, w = img_bgr.shape[:2]
    fx, fy, fw, fh = map(int, face_row[:4])
    px, py, size, _ = compute_square_face_box(fx, fy, fw, fh, w, h, [float(v) for v in face_row[4:8]])
    return img_bgr[py:py + size, px:px + size]


def clip_image_embedding(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    session = get_clip_session()
    if session is None:
        return None
    x = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255.0
    x = ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None]
    v = session.run(None, {"pixel_values": x})[0][0]
    return v / np.linalg.norm(v)


def visual_embedding(img_bgr: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
    return clip_image_embedding(head_crop(img_bgr, face_row))


# ArcFace cosine -> human-facing percentage, piecewise-linear between anchors.
# Anchors come from tools/calibrate_scores.py (2026-09 run, single photo vs single photo):
#   strangers (3000 pairs)   p50 0.008  p90 0.091  p95 0.114
#   kin (740 real pairs)     p25 0.076  p50 0.136  p75 0.204  p90 0.285  p95 0.351
#   same person (567 pairs)  p5 0.325   p10 0.389  p25 0.473  p50 0.567
#   celebrity top-1 vs DB    p5 0.276   p25 0.328  p50 0.360  p90 0.421  p95 0.442  (3,358-person DB)
SIMILARITY_ANCHORS = {
    # Two photos, "how alike do these two people look". Verdicts: >=82 판박이, >=68 매우 높음, >=50 은근한, >=35 낮음.
    # stranger median -> ~20 (남남), kin median -> ~54 (은근한), kin p75 -> ~68, kin p90 -> ~81, same person -> 95+
    "family": [(-1.0, 3), (-0.05, 10), (0.0, 18), (0.05, 30), (0.10, 42), (0.14, 55), (0.20, 68),
               (0.28, 80), (0.35, 87), (0.45, 92), (0.60, 96), (1.0, 99)],
    # Two photos, "is this the same person". Verdicts: >=85 확실, >=70 유력, >=50 가능성 낮음.
    # kin p95 -> ~63 (not "유력"), same-person p10 -> ~71, same-person p25 -> ~85
    "identical": [(-1.0, 1), (0.0, 3), (0.15, 15), (0.25, 35), (0.32, 55), (0.38, 70), (0.47, 85),
                  (0.60, 94), (0.80, 98), (1.0, 99)],
    # One photo vs a celebrity's multi-photo mean. Aligned with CELEB_TOP1_PERCENTILES so the % and the
    # "match strength" label agree: p10 -> 60, p50 -> 72, p90 -> 84, p95 -> 87, same person -> 95+
    "celebrity": [(-1.0, 5), (0.0, 20), (0.15, 40), (0.239, 52), (0.297, 60), (0.328, 66), (0.36, 72),
                  (0.391, 78), (0.421, 84), (0.442, 87), (0.50, 91), (0.60, 95), (1.0, 99)],
}


# Best-match cosine of other people against the celebrity DB (3,358 people, celebrity photos, self excluded).
# (cosine, percentile) pairs; used to tell users how strong their #1 match is compared with everyone else's #1.
CELEB_TOP1_PERCENTILES = [(0.239, 1), (0.276, 5), (0.297, 10), (0.328, 25), (0.360, 50),
                          (0.391, 75), (0.421, 90), (0.442, 95), (0.575, 99)]


def celebrity_match_strength(cosine: float) -> Dict[str, Any]:
    """Percentile of a #1 match among typical #1 matches, with a short honest Korean label."""
    xs, ys = zip(*CELEB_TOP1_PERCENTILES)
    pct = float(np.clip(np.interp(cosine, xs, ys), 0.5, 99.5))
    top = max(1, int(round(100 - pct)))
    if pct >= 90:
        label, detail = "아주 뚜렷한 닮은꼴", f"다른 사람들의 1위 닮은꼴과 비교해 상위 {top}%예요."
    elif pct >= 60:
        label, detail = "평균보다 닮은 편", f"다른 사람들의 1위 닮은꼴과 비교해 상위 {top}%예요."
    elif pct >= 30:
        label, detail = "보통 수준의 닮음", "다른 사람들이 찾은 1위 닮은꼴과 비슷한 수준이에요."
    else:
        label, detail = "약한 닮음", "크게 닮은 연예인은 없어서, 가장 가까운 후보를 보여 드려요."
    return {"percentile": round(pct, 1), "label": label, "detail": detail}


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
