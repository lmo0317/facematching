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
import urllib.request
import urllib.parse
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


def download_wikipedia_portrait(query: str, lang: str, out_path: str, timeout: float = 8) -> bool:
    """
    Search Wikipedia for `query`, download the page's lead image, square-crop it
    (biased toward the top third where faces usually are) to 600x600 and save as JPEG.
    Returns False when no image is found; network/decoding errors propagate.
    """
    encoded = urllib.parse.quote(query)
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}"
        "&gsrlimit=1&prop=pageimages&piprop=original&format=json"
    )
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))

    img_url = None
    for info in data.get("query", {}).get("pages", {}).values():
        if "original" in info and "source" in info["original"]:
            img_url = info["original"]["source"]
            break
    if not img_url:
        return False

    img_req = urllib.request.Request(img_url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(img_req, timeout=timeout + 4) as ir:
        raw_bytes = ir.read()

    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = max(0, (h - min_dim) // 3)
        if top + min_dim > h:
            top = h - min_dim
        cropped = img.crop((left, top, left + min_dim, top + min_dim))
        cropped.resize((600, 600), Image.Resampling.LANCZOS).save(out_path, format="JPEG", quality=85, optimize=True)

    return True
