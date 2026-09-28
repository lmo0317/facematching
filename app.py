import os
import io
import base64
import logging
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import httpx
import cv2
import numpy as np

from face_utils import (
    YUNET_PATH, compute_square_face_box, image_bytes_to_data_url, extract_json_block, strip_thought_tags,
    extract_face_embedding, similarity_to_percent, adjust_part_scores, data_url_to_bytes,
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("facematch")

# Environment configurations
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-e4b-it-q4km")
PORT = int(os.getenv("PORT", "8501"))
HOST = os.getenv("HOST", "0.0.0.0")

app = FastAPI(
    title="FaceMatch AI - Gemma 4 E4B",
    description="사진 2장을 정밀 대조하여 유사도 및 얼굴 특징을 분석하는 웹 애플리케이션",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Mount static folder (supports both /static and /facematching/static)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/facematching/static", StaticFiles(directory=STATIC_DIR), name="static_facematching")


def face_position_label(index: int, count: int) -> str:
    """Korean label for a face sorted left-to-right, e.g. '인물 1 (왼쪽)'."""
    if count == 2:
        return f"인물 {index + 1} ({['왼쪽', '오른쪽'][index]})"
    if count == 3:
        return f"인물 {index + 1} ({['왼쪽', '중앙', '오른쪽'][index]})"
    return f"인물 {index + 1}"


def detect_faces_in_image(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Detect faces using YuNet (or Haar Cascade fallback) and return balanced 1:1 coordinates + thumbnails."""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes))
        img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        w, h = img_pil.size
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces = None
        if os.path.exists(YUNET_PATH):
            try:
                detector = cv2.FaceDetectorYN.create(
                    model=YUNET_PATH,
                    config="",
                    input_size=(w, h),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=50
                )
                detector.setInputSize((w, h))
                _, faces = detector.detect(img_bgr)
            except Exception as e:
                logger.warning(f"YuNet detection warning: {e}")
                faces = None

        # Normalize detector output to (x, y, w, h, confidence, landmarks)
        raw_faces: List[Tuple[int, int, int, int, float, Optional[List[float]]]] = []
        if faces is not None and len(faces) > 0:
            for f in faces:
                landmarks = [float(x) for x in f[4:14]] if len(f) >= 14 else None
                raw_faces.append((*map(int, f[0:4]), float(f[-1]), landmarks))
        else:
            # Fallback to Haar Cascade
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                for (fx, fy, fw, fh) in haar_faces:
                    raw_faces.append((int(fx), int(fy), int(fw), int(fh), 0.85, None))
            except Exception as e:
                logger.warning(f"Haar Cascade fallback warning: {e}")

        raw_faces.sort(key=lambda x: x[0])
        count = len(raw_faces)
        detected = []
        for i, (fx, fy, fw, fh, score, landmarks) in enumerate(raw_faces):
            px, py, pw, ph = compute_square_face_box(fx, fy, fw, fh, w, h, landmarks)

            thumb = img_pil.crop((px, py, px + pw, py + ph))
            thumb = thumb.resize((80, 80), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            detected.append({
                "id": i + 1,
                "label": face_position_label(i, count),
                "box": {"x": fx, "y": fy, "width": fw, "height": fh},
                "padded_box": {"x": px, "y": py, "width": pw, "height": ph},
                "confidence": round(score, 2),
                "thumbnail": thumb_b64
            })

        return detected
    except Exception as e:
        logger.error(f"Face detection error: {e}", exc_info=True)
        return []


def process_image_to_base64(image_bytes: bytes, max_dim: int = 768) -> str:
    """Normalize an uploaded image into a JPEG data URL, mapping failures to HTTP 400."""
    try:
        return image_bytes_to_data_url(image_bytes, max_dim)
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Robust extraction of JSON from model response text, with a low-score fallback structure."""
    parsed = extract_json_block(text)
    if parsed is not None:
        return parsed

    logger.warning("Failed to parse strict JSON, building fallback response from raw text")
    return {
        "similarity_score": 30,
        "verdict": "서로 다른 생김새",
        "verdict_summary": "두 사람의 이목구비 형태 대조가 완료되었습니다.",
        "detailed_scores": {
            "face_shape": 30,
            "eyes": 30,
            "nose": 30,
            "mouth": 30,
            "features": 30
        },
        "similarities": ["세부 내용은 종합 분석을 확인하세요."],
        "differences": ["세부 내용은 종합 분석을 확인하세요."],
        "environmental_factors": "나이, 성별, 조명 및 각도 차이를 감안하여 분석함",
        "comprehensive_analysis": strip_thought_tags(text).strip()
    }


def calibrate_similarity_score(raw_score: float) -> int:
    """
    Calibrate raw LLM vision score to human-perceived facial resemblance scale:
    - Raw <= 40: Strangers -> 10% ~ 25% (서로 다른 생김새)
    - Raw 40 ~ 50: Weak resemblance -> 25% ~ 45% (서로 다른 인물 / 남남)
    - Raw 50 ~ 65: Moderate / Family -> 45% ~ 76% (은근한 닮음 ~ 높은 붕어빵)
    - Raw 65 ~ 85: Strong Family / Lookalike -> 76% ~ 92% (완벽한 붕어빵)
    - Raw > 85: Identical / Twin -> 92% ~ 99% (도플갱어 / 동일인 수준)
    """
    raw = float(raw_score)
    if raw <= 40:
        return max(10, int(round(10 + (raw / 40.0) * 15)))
    elif raw <= 50:
        return int(round(25 + ((raw - 40.0) / 10.0) * 20))
    elif raw <= 65:
        return int(round(45 + ((raw - 50.0) / 15.0) * 31))
    elif raw <= 85:
        return int(round(76 + ((raw - 65.0) / 20.0) * 16))
    else:
        return min(99, int(round(92 + ((raw - 85.0) / 15.0) * 7)))


def get_verdict_for_score(score: int, mode: str = "family") -> str:
    if mode == "identical":
        if score >= 85:
            return "동일 인물 확실"
        elif score >= 70:
            return "동일 인물 유력"
        elif score >= 50:
            return "동일인 가능성 낮음"
        else:
            return "다른 인물 (불일치)"
    elif mode == "celebrity":
        if score >= 85:
            return "도플갱어 수준(판박이)"
        elif score >= 70:
            return "매우 높은 닮은꼴"
        elif score >= 50:
            return "은근한 분위기 닮은꼴"
        elif score >= 35:
            return "부분적 닮음"
        else:
            return "서로 다른 생김새"
    else:  # family
        if score >= 82:
            return "완벽한 붕어빵 (판박이)"
        elif score >= 68:
            return "매우 높은 붕어빵 지수"
        elif score >= 50:
            return "은근한 닮음 / 부분 닮음"
        elif score >= 35:
            return "낮은 닮음 (미미한 유사성)"
        else:
            return "서로 다른 생김새 (남남)"


class CompareJsonRequest(BaseModel):
    image1_base64: str
    image2_base64: str
    mode: Optional[str] = "family"


class DetectFacesRequest(BaseModel):
    image_base64: str


class CelebrityMatchRequest(BaseModel):
    image_base64: str
    gender_filter: Optional[str] = "auto"


@app.get("/facematching")
async def redirect_facematching():
    return RedirectResponse(url="/facematching/", status_code=302)


@app.get("/", response_class=HTMLResponse)
@app.get("/facematching/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FaceMatch AI - 로딩 중...</h1>")


@app.get("/api/health")
@app.get("/facematching/api/health")
async def health_check():
    """Check connectivity to Gemma 4 llama-server."""
    backend_status = "unknown"
    model_found = False
    models_list = []
    has_multimodal = False

    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{LLAMA_SERVER_URL}/v1/models")
            if resp.status_code == 200:
                backend_status = "connected"
                data = resp.json()
                models_data = data.get("models", []) or data.get("data", [])
                for m in models_data:
                    m_id = m.get("id") or m.get("name") or ""
                    models_list.append(m_id)
                    caps = m.get("capabilities", [])
                    if "multimodal" in caps or "vision" in caps:
                        has_multimodal = True
                    if MODEL_NAME in m_id or "gemma-4" in m_id:
                        model_found = True
            else:
                backend_status = f"http_{resp.status_code}"
    except Exception as e:
        backend_status = f"unreachable: {str(e)}"

    return {
        "status": "ok",
        "backend": {
            "url": LLAMA_SERVER_URL,
            "status": backend_status,
            "target_model": MODEL_NAME,
            "model_found": model_found,
            "available_models": models_list,
            "multimodal_enabled": has_multimodal
        }
    }


@app.get("/api/samples")
@app.get("/facematching/api/samples")
async def get_samples():
    """List sample image pairs available for instant testing."""
    sample_pairs = [
        {
            "id": "pair1",
            "title": "예제 1 (부자 붕어빵 - 아빠 & 아들)",
            "description": "아빠와 아들의 눈매, 콧날, 웃는 입모양 유전적 붕어빵 싱크로율 대조",
            "expected": "완벽한 붕어빵 (85% 이상)",
            "mode": "family",
            "img1": "/static/samples/sample_father.jpg",
            "img2": "/static/samples/sample_son.jpg"
        },
        {
            "id": "pair2",
            "title": "예제 2 (모자 붕어빵 - 엄마 & 아들)",
            "description": "성별 차이를 보정한 엄마와 아들의 선한 눈매와 온화한 미소 대조",
            "expected": "완벽한 붕어빵 (85% 이상)",
            "mode": "family",
            "img1": "/static/samples/sample_mother.jpg",
            "img2": "/static/samples/sample_mother_son.jpg"
        },
        {
            "id": "pair3",
            "title": "예제 3 (서로 다른 생김새 - 남남 비교)",
            "description": "혈연 관계가 없는 서로 다른 두 인물의 이목구비 대조",
            "expected": "서로 다른 생김새 (35% 이하)",
            "mode": "family",
            "img1": "/static/samples/sample2_a.jpg",
            "img2": "/static/samples/sample2_b.jpg"
        },
        {
            "id": "pair4",
            "title": "예제 4 (단체 사진 인물 선택 - 3인 가족 중 아들 vs 엄마)",
            "description": "3인 가족 단체 사진에서 원하는 인물을 클릭하여 맞춘 뒤 붕어빵 대조",
            "expected": "완벽한 붕어빵 (80% 이상)",
            "mode": "family",
            "img1": "/static/samples/sample_family_group.jpg",
            "img2": "/static/samples/sample_mother.jpg"
        }
    ]
    return {"samples": sample_pairs}


@app.post("/api/detect-faces")
@app.post("/facematching/api/detect-faces")
async def detect_faces_endpoint(req: DetectFacesRequest):
    """Detect human faces in image, returning bounding boxes, labels, and thumbnails."""
    data = req.image_base64
    if "," in data:
        data = data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="유효한 Base64 이미지가 아닙니다.")

    faces = detect_faces_in_image(image_bytes)
    return {
        "success": True,
        "count": len(faces),
        "faces": faces
    }


@app.post("/api/compare")
@app.post("/facematching/api/compare")
async def compare_images(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    mode: Optional[str] = Form("family")
):
    """Multipart upload comparison endpoint."""
    if not image1 or not image2:
        raise HTTPException(status_code=400, detail="사진 2장을 모두 업로드해 주세요.")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    data_url1 = process_image_to_base64(bytes1)
    data_url2 = process_image_to_base64(bytes2)

    return await execute_face_comparison(data_url1, data_url2, mode or "family")


@app.post("/api/compare-json")
@app.post("/facematching/api/compare-json")
async def compare_images_json(req: CompareJsonRequest):
    """JSON base64 comparison endpoint."""
    # Ensure data url format
    img1 = req.image1_base64
    img2 = req.image2_base64

    # If raw base64, prepend data:image/jpeg;base64,
    if not img1.startswith("data:"):
        img1 = f"data:image/jpeg;base64,{img1}"
    if not img2.startswith("data:"):
        img2 = f"data:image/jpeg;base64,{img2}"

    return await execute_face_comparison(img1, img2, req.mode or "family")


@app.post("/api/find-celebrity")
@app.post("/facematching/api/find-celebrity")
async def find_celebrity_endpoint(req: CelebrityMatchRequest):
    """Analyze single user photo and find TOP 3 celebrity lookalikes with real photos."""
    img = req.image_base64
    if not img.startswith("data:"):
        img = f"data:image/jpeg;base64,{img}"

    from celebrity_service import execute_celebrity_lookalike, FaceNotFoundError
    try:
        result = await execute_celebrity_lookalike(
            image_b64=img,
            gender_filter=req.gender_filter or "auto",
            llama_url=LLAMA_SERVER_URL,
            model_name=MODEL_NAME
        )
        return result
    except FaceNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Celebrity match error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"닮은 연예인 검색 중 오류 발생: {str(e)}")


PART_KEYS = ["eyes", "nose", "mouth", "face_shape", "features"]
# Share of the final score taken from the ArcFace measurement (rest: Gemma's visual judgement).
# Gemma alone could not separate family from strangers in testing, so the measurement dominates.
FACE_SCORE_WEIGHT = {"family": 0.8, "celebrity": 0.8, "identical": 1.0}


async def execute_face_comparison(img1_url: str, img2_url: str, mode: str = "family") -> Dict[str, Any]:
    """ArcFace similarity (primary score) + Gemma 4 visual analysis (per-part pattern and written report)."""
    emb1 = extract_face_embedding(data_url_to_bytes(img1_url))
    emb2 = extract_face_embedding(data_url_to_bytes(img2_url))
    cosine = float(emb1 @ emb2) if emb1 is not None and emb2 is not None else None
    face_score = similarity_to_percent(cosine, "identical" if mode == "identical" else "family") if cosine is not None else None

    llm_ok = True
    try:
        analysis, raw_text = await request_gemma_comparison(img1_url, img2_url, mode, face_score)
    except HTTPException as e:
        if face_score is None:
            raise
        logger.warning(f"Gemma unavailable ({e.detail}); returning face-recognition-only result")
        llm_ok, raw_text = False, None
        analysis = {
            "detailed_scores": {k: face_score for k in PART_KEYS},
            "similarity_score": face_score,
            "verdict_summary": f"얼굴 인식 AI가 측정한 두 얼굴의 골격·이목구비 유사도는 {face_score}%입니다.",
            "similarities": [],
            "differences": [],
            "environmental_factors": "AI 소견 서버가 응답하지 않아 얼굴 인식 측정값만 표시합니다.",
            "comprehensive_analysis": "상세 소견을 생성하지 못했습니다. 잠시 후 다시 시도하면 부위별 설명을 볼 수 있습니다.",
        }

    llm_score = analysis["similarity_score"]
    if face_score is not None:
        weight = FACE_SCORE_WEIGHT.get(mode, 0.8) if llm_ok else 1.0
        final_score = int(round(weight * face_score + (1 - weight) * llm_score))
        analysis["detailed_scores"] = adjust_part_scores(analysis["detailed_scores"], final_score, PART_KEYS)
    else:
        weight = 0.0
        final_score = llm_score
        analysis["environmental_factors"] = (analysis.get("environmental_factors") or "") + \
            " (한쪽 사진에서 얼굴을 인식하지 못해 AI 시각 판단만으로 점수를 산출했습니다.)"

    analysis["similarity_score"] = final_score
    analysis["verdict"] = get_verdict_for_score(final_score, mode)
    analysis["score_breakdown"] = {
        "face_recognition": face_score,
        "cosine": round(cosine, 4) if cosine is not None else None,
        "llm": llm_score if llm_ok else None,
        "face_weight": weight,
    }
    return {
        "success": True,
        "model": MODEL_NAME,
        "data": analysis,
        "raw_output": raw_text if raw_text and logger.isEnabledFor(logging.DEBUG) else None
    }


async def request_gemma_comparison(img1_url: str, img2_url: str, mode: str,
                                   face_score: Optional[int]) -> Tuple[Dict[str, Any], str]:
    """Gemma 4 E4B visual comparison. Returns (analysis with calibrated LLM scores, raw text); raises HTTPException."""
    
    if mode == "identical":
        mode_title = "동일 인물 정밀 대조 (신원 확인)"
        mode_instruction = (
            "【분석 모드: 동일 인물 정밀 대조 (신원 확인)】\n"
            "목적: 제시된 [사진 1]과 [사진 2]가 실제 '동일한 한 사람'의 사진인지 과학적이고 엄격하게 판별합니다.\n"
            "판정 가이드라인:\n"
            "- 성별, 안면 영구 골격 비율, 귀 모양, 눈코입의 절대적 비례를 정밀 분석합니다.\n"
            "- 헤어스타일, 안경, 메이크업의 일시적 변화에 속지 말고 본질적 골격 구조만 비교하세요.\n"
            "- 기준: 동일 인물이면 85~100점, 일부 표정 차이 유력은 70~84점, 다른 인물이면 10~35점.\n"
        )
    elif mode == "celebrity":
        mode_title = "닮은꼴 싱크로율 (연예인·친구 닮은꼴 측정)"
        mode_instruction = (
            "【분석 모드: 닮은꼴 싱크로율 (친구·연예인 닮은꼴 측정)】\n"
            "목적: 서로 다른 두 사람의 생김새와 인상이 얼마나 닮았는지 닮은꼴 싱크로율을 냉정하게 측정합니다.\n"
            "판정 가이드라인:\n"
            "- 안경 착용, 헤어스타일, 옷차림, 단순히 둘 다 웃고 있다는 표정 연출에 속지 마세요!\n"
            "- 전혀 닮지 않은 타인/남남은 15~35점의 '서로 다른 생김새' 점수를 단호하게 부여하세요.\n"
            "- 특정 이목구비 하나만 얼핏 닮았으면 45~60점, 한눈에 알아볼 만큼 닮았으면 70~85점, 도플갱어 수준이면 86~98점을 부여하세요.\n"
        )
    else:  # default: family
        mode_title = "가족·붕어빵 닮음도 분석 (부자/모자/형제자매)"
        mode_instruction = (
            "【분석 모드: 가족·붕어빵 닮음도 분석 (아빠-아들, 엄마-아들/딸, 형제·자매 등)】\n"
            "목적: 두 사람 간의 유전적 생김새 닮음도(붕어빵 지수)를 과학적이고 객관적으로 측정합니다.\n\n"
            "★★★★★ [판정 및 채점 가이드라인 - 핵심 원칙] ★★★★★\n"
            "1. [외적 연출 배제 (절대 속지 마십시오)]:\n"
            "   - 안경 착용 여부, 비슷한 헤어스타일, 옷차림, 단순히 둘 다 웃고 있다는 표정에 속지 마십시오.\n"
            "   - 안경을 둘 다 썼거나 둘 다 웃는다고 해서 닮은 것이 아닙니다. 오직 얼굴 골격과 이목구비 자체의 순수 형태만 대조하십시오.\n\n"
            "2. [전혀 다른 남남에 대한 단호한 감점 (맹목적 고득점 절대 금지)]:\n"
            "   - 두 사람이 유전적 연관이 없는 타인/남남인 경우:\n"
            "   - 억지로 공통점을 지어내어 60~80점 이상의 높은 점수를 주어서는 절대 안 됩니다!\n"
            "   - 턱선, 눈매, 코 모양, 입술 등 이목구비 형태가 상이하다면 가차 없이 15점 ~ 38점 사이의 '서로 다른 생김새' 점수를 부여하십시오!\n\n"
            "3. [실제 붕어빵 가족에 대한 정당한 점수 부여]:\n"
            "   - 부모와 자식 간 20~30년의 나이 차이로 인한 자연스러운 노화(주름, 피부 처짐, 흰머리)는 감점 요인이 아닙니다.\n"
            "   - 엄마-아들처럼 성별이 달라도, 엄마의 선한 눈매와 웃는 입매를 아들이 그대로 물려받았는지를 확인하십시오.\n"
            "   - 주름을 지우고 타고난 이목구비(쌍꺼풀 라인, 콧날 각도, 미소 시 입꼬리 굴곡, 턱 끝 형태)가 쏙 빼닮았다면 75점 ~ 90점의 높은 붕어빵 점수를 부여하십시오.\n\n"
            "4. [5개 부위별 엄격한 채점 기준 (각 0~100점)]:\n"
            "   - eyes (눈매): 쌍꺼풀 라인, 눈의 가로세로 비례, 눈꼬리 각도, 눈웃음 모양\n"
            "     (붕어빵: 75~95점 | 부분 유사: 45~65점 | 남남: 15~35점)\n"
            "   - nose (콧대/콧볼): 콧대 높이, 콧볼 너비(복코 vs 날렵한 코), 코끝 형태\n"
            "     (붕어빵: 75~95점 | 부분 유사: 45~65점 | 남남: 15~35점)\n"
            "   - mouth (입술/미소): 웃을 때 입꼬리 굴곡, 치아 노출, 입술 두께와 하관 인상\n"
            "     (붕어빵: 75~95점 | 부분 유사: 45~65점 | 남남: 15~35점)\n"
            "   - face_shape (얼굴형): 턱끝 형태, 하악각 윤곽선\n"
            "     (붕어빵: 70~90점 | 부분 유사: 45~65점 | 남남: 15~35점)\n"
            "   - features (고유 인상): 이목구비 상대적 배치 비율 및 유전적 싱크로율\n"
            "     (붕어빵: 75~95점 | 부분 유사: 45~65점 | 남남: 15~35점)\n"
        )

    reference = ""
    if face_score is not None:
        reference = (
            f"【참고 측정값】: 얼굴 인식 AI(ArcFace)가 두 얼굴의 골격·이목구비 구조를 수치로 비교한 유사도는 {face_score}%이며, "
            f"이에 따른 판정은 '{get_verdict_for_score(face_score, mode)}'입니다.\n"
            "verdict_summary와 comprehensive_analysis는 반드시 이 판정과 같은 결론으로 작성하세요 "
            "(판정보다 약하게 '분위기만 비슷하다'거나 더 강하게 쓰지 마세요). "
            "그 결론을 뒷받침하는 공통점과 차이점을 사진에서 찾아 설명하고, 부위별 점수는 직접 관찰해 부위 간 차이가 드러나게 매기세요.\n\n"
        )

    system_instruction = (
        f"{mode_instruction}\n\n{reference}"
        "반드시 다음 JSON 스키마 형식으로만 출력하세요 (마크다운 코드블록이나 서두/결말 문장 없이 순수한 JSON 문자열만 출력):\n"
        "{\n"
        '  "detailed_scores": {\n'
        '    "eyes": <눈매 점수 0-100>,\n'
        '    "nose": <콧대 점수 0-100>,\n'
        '    "mouth": <입술 점수 0-100>,\n'
        '    "face_shape": <얼굴형 점수 0-100>,\n'
        '    "features": <고유 인상 점수 0-100>\n'
        '  },\n'
        '  "similarity_score": <부위별 점수를 종합한 정수 0-100>,\n'
        '  "verdict_summary": "<두 사람의 닮음 정도를 친절하고 명확하게 설명하는 한 문장>",\n'
        '  "similarities": [\n'
        '    "<실제 관찰된 구체적 붕어빵 포인트 1~2개>"\n'
        '  ],\n'
        '  "differences": [\n'
        '    "<명백하게 다른 부위별 형태 차이점 1~2개>"\n'
        '  ],\n'
        '  "environmental_factors": "<나이, 성별, 표정, 안경 등 감안하거나 배제한 요소>",\n'
        '  "comprehensive_analysis": "<종합 분석 소견 2~3문장>"\n'
        "}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"다음 두 사람의 얼굴을 비교하여 얼마나 닮았는지 붕어빵 싱크로율을 감정해 주세요. [모드: {mode_title}]\n\n[사진 1]:"},
                {"type": "image_url", "image_url": {"url": img1_url}},
                {"type": "text", "text": "\n[사진 2]:"},
                {"type": "image_url", "image_url": {"url": img2_url}},
                {"type": "text", "text": f"\n\n{system_instruction}"}
            ]
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.05,
        "max_tokens": 1500,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if resp.status_code != 200:
                err_text = resp.text
                logger.error(f"LLM API Error ({resp.status_code}): {err_text}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemma 4 모델 서버 응답 오류 ({resp.status_code}): {err_text}"
                )

            result_data = resp.json()
            choices = result_data.get("choices", [])
            if not choices:
                raise HTTPException(status_code=502, detail="모델에서 유효한 응답을 생성하지 못했습니다.")

            content_text = choices[0].get("message", {}).get("content", "")
            parsed_analysis = extract_json_from_response(content_text)

            # Calibrate Gemma's raw scores into the human-perceived scale
            det = parsed_analysis.get("detailed_scores", {})
            raw_parts = {}
            for k in PART_KEYS:
                try:
                    raw_parts[k] = float(det.get(k, 40))
                except (TypeError, ValueError):
                    raw_parts[k] = 40.0
            parsed_analysis["detailed_scores"] = {k: calibrate_similarity_score(v) for k, v in raw_parts.items()}

            try:
                raw_llm = float(parsed_analysis.get("similarity_score", 40))
            except (TypeError, ValueError):
                raw_llm = 40.0
            raw_weighted = (raw_parts["eyes"] * 0.25 + raw_parts["nose"] * 0.20 + raw_parts["mouth"] * 0.20
                            + raw_parts["face_shape"] * 0.20 + raw_parts["features"] * 0.15)
            parsed_analysis["similarity_score"] = calibrate_similarity_score(0.4 * raw_llm + 0.6 * raw_weighted)

            parsed_analysis.setdefault("verdict_summary", "분석이 완료되었습니다.")
            parsed_analysis.setdefault("similarities", [])
            parsed_analysis.setdefault("differences", [])
            parsed_analysis.setdefault("environmental_factors", "특이사항 없음")
            parsed_analysis.setdefault("comprehensive_analysis", content_text)
            return parsed_analysis, content_text

    except httpx.TimeoutException:
        logger.error("LLM Server timeout")
        raise HTTPException(status_code=504, detail="Gemma 4 모델 추론 시간 초과 (90초). 잠시 후 다시 시도해 주세요.")
    except httpx.ConnectError:
        logger.error("LLM Server connection failed")
        raise HTTPException(
            status_code=503,
            detail=f"Gemma 4 서버({LLAMA_SERVER_URL})에 연결할 수 없습니다. 서비스가 실행 중인지 확인하세요."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"분석 중 내부 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
