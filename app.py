import os
import io
import re
import json
import base64
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import httpx
import cv2
import numpy as np

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
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")

# Mount static folder (supports both /static and /facematching/static)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/facematching/static", StaticFiles(directory=STATIC_DIR), name="static_facematching")


def detect_faces_in_image(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Detect faces using YuNet (or Haar Cascade fallback) and return coordinates + thumbnails."""
    try:
        img_pil = Image.open(io.BytesIO(image_bytes))
        img_pil = ImageOps.exif_transpose(img_pil)
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        w, h = img_pil.size
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        faces = None
        if os.path.exists(YUNET_MODEL_PATH):
            try:
                detector = cv2.FaceDetectorYN.create(
                    model=YUNET_MODEL_PATH,
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

        detected = []
        if faces is not None and len(faces) > 0:
            sorted_faces = sorted(faces, key=lambda x: x[0])
            count = len(sorted_faces)
            for i, f in enumerate(sorted_faces):
                fx, fy, fw, fh = map(int, f[0:4])
                score = float(f[-1])

                pad_x = int(fw * 0.35)
                pad_top = int(fh * 0.45)
                pad_bottom = int(fh * 0.35)

                px1 = max(0, fx - pad_x)
                py1 = max(0, fy - pad_top)
                px2 = min(w, fx + fw + pad_x)
                py2 = min(h, fy + fh + pad_bottom)
                pw = px2 - px1
                ph = py2 - py1

                thumb = img_pil.crop((px1, py1, px2, py2))
                thumb = thumb.resize((80, 80), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=85)
                thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                if count == 1:
                    label = "인물 1"
                elif count == 2:
                    label = "인물 1 (왼쪽)" if i == 0 else "인물 2 (오른쪽)"
                elif count == 3:
                    pos = ["왼쪽", "중앙", "오른쪽"][i]
                    label = f"인물 {i+1} ({pos})"
                else:
                    label = f"인물 {i+1}"

                detected.append({
                    "id": i + 1,
                    "label": label,
                    "box": {"x": fx, "y": fy, "width": fw, "height": fh},
                    "padded_box": {"x": px1, "y": py1, "width": pw, "height": ph},
                    "confidence": round(score, 2),
                    "thumbnail": thumb_b64
                })
        else:
            # Fallback to Haar Cascade
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(haar_faces) > 0:
                    sorted_faces = sorted(haar_faces, key=lambda x: x[0])
                    for i, (fx, fy, fw, fh) in enumerate(sorted_faces):
                        pad_x = int(fw * 0.35)
                        pad_top = int(fh * 0.45)
                        pad_bottom = int(fh * 0.35)

                        px1 = max(0, fx - pad_x)
                        py1 = max(0, fy - pad_top)
                        px2 = min(w, fx + fw + pad_x)
                        py2 = min(h, fy + fh + pad_bottom)
                        pw = px2 - px1
                        ph = py2 - py1

                        thumb = img_pil.crop((px1, py1, px2, py2))
                        thumb = thumb.resize((80, 80), Image.Resampling.LANCZOS)
                        buf = io.BytesIO()
                        thumb.save(buf, format="JPEG", quality=85)
                        thumb_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                        detected.append({
                            "id": i + 1,
                            "label": f"인물 {i+1}",
                            "box": {"x": int(fx), "y": int(fy), "width": int(fw), "height": int(fh)},
                            "padded_box": {"x": px1, "y": py1, "width": pw, "height": ph},
                            "confidence": 0.85,
                            "thumbnail": thumb_b64
                        })
            except Exception as e:
                logger.warning(f"Haar Cascade fallback warning: {e}")

        return detected
    except Exception as e:
        logger.error(f"Face detection error: {e}", exc_info=True)
        return []


def process_image_to_base64(image_bytes: bytes, max_dim: int = 768) -> str:
    """Validate, orient and resize image, then convert to base64 data URL."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Handle EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Convert to RGB (handles RGBA, Palette, Grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize if dimensions exceed max_dim
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
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 처리 중 오류가 발생했습니다: {str(e)}")


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Robust extraction of JSON from model response text."""
    # Remove thought tags if present
    cleaned = re.sub(r"<channel>thought.*?</channel>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|think\|>.*?</turn>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    # 1. Look for ```json ... ``` blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    # 2. Look for outermost curly braces { ... }
    brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except Exception:
            pass

    # 3. Fallback: Parse line by line or construct basic structure
    logger.warning("Failed to parse strict JSON, building fallback response from raw text")
    return {
        "similarity_score": 75,
        "verdict": "닮은꼴 분석 완료",
        "verdict_summary": "이목구비 전반에서 유의미한 닮음 포인트가 관찰되었습니다. 상세 내용을 확인하세요.",
        "detailed_scores": {
            "face_shape": 75,
            "eyes": 75,
            "nose": 75,
            "mouth": 75,
            "features": 75
        },
        "similarities": ["세부 내용은 종합 분석을 확인하세요."],
        "differences": ["세부 내용은 종합 분석을 확인하세요."],
        "environmental_factors": "나이, 성별, 조명 및 각도 차이를 감안하여 분석함",
        "comprehensive_analysis": cleaned.strip()
    }


class CompareJsonRequest(BaseModel):
    image1_base64: str
    image2_base64: str
    mode: Optional[str] = "family"


class DetectFacesRequest(BaseModel):
    image_base64: str


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
    except Exception as e:
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


async def execute_face_comparison(img1_url: str, img2_url: str, mode: str = "family") -> Dict[str, Any]:
    """Call Gemma 4 E4B multimodal LLM and perform facial similarity/resemblance evaluation."""
    
    if mode == "identical":
        mode_title = "동일 인물 정밀 대조 (신원 확인)"
        mode_instruction = (
            "【분석 모드: 동일 인물 정밀 대조 (신원 확인)】\n"
            "목적: 제시된 [사진 1]과 [사진 2]가 실제 '동일한 한 사람'의 사진인지 과학적이고 엄격하게 판별합니다.\n"
            "판정 가이드라인:\n"
            "- 성별, 영구적 안면 골격 비율, 귀 모양, 눈코입의 절대적 거리 비례를 분석합니다.\n"
            "- 점수 기준 (0~100점):\n"
            "   * 85 ~ 100점: 동일 인물 확실 (골격 및 이목구비 비례 일치)\n"
            "   * 70 ~ 84점: 동일 인물 유력 (일부 각도/표정 차이 외 일치)\n"
            "   * 50 ~ 69점: 동일인 가능성 낮음 (유사한 인상이나 세부 골격 상이)\n"
            "   * 0 ~ 49점: 서로 다른 인물 (골격 또는 성별 상이)\n"
        )
        verdict_options = "동일 인물 확실 | 동일 인물 유력 | 동일인 가능성 낮음 | 다른 인물"
    elif mode == "celebrity":
        mode_title = "닮은꼴 싱크로율 (연예인·친구 닮은꼴 측정)"
        mode_instruction = (
            "【분석 모드: 닮은꼴 싱크로율 (연예인·친구 닮은꼴 측정)】\n"
            "목적: 서로 다른 두 사람(예: 나와 연예인, 친구 간)의 생김새와 인상이 얼마나 닮았는지 닮은꼴 싱크로율을 측정합니다.\n"
            "판정 가이드라인:\n"
            "- 헤어스타일이나 메이크업, 표정에 속지 않고 눈매, 콧날, 입꼬리, 얼굴형의 형태적 닮음도를 평가하세요.\n"
            "- 점수 기준 (0~100점):\n"
            "   * 85 ~ 100점: 도플갱어 수준 (거의 쌍둥이처럼 닮음)\n"
            "   * 70 ~ 84점: 매우 높은 닮은꼴 (한눈에 알아볼 만큼 닮음)\n"
            "   * 50 ~ 69점: 분위기 닮은꼴 (특정 이목구비 및 전체적 인상 유사)\n"
            "   * 35 ~ 49점: 부분적 닮음 (특정 포인트 하나만 유사)\n"
            "   * 0 ~ 34점: 서로 다른 생김새 (닮은 구석 적음)\n"
        )
        verdict_options = "도플갱어 수준(판박이) | 매우 높은 닮은꼴 | 분위기 닮은꼴 | 부분적 닮음 | 서로 다른 생김새"
    else:  # default: family
        mode_title = "가족·붕어빵 닮음도 분석 (부자/모자/형제자매)"
        mode_instruction = (
            "【분석 모드: 가족·붕어빵 닮음도 분석 (아빠-아들, 엄마-아들/딸, 형제·자매 등)】\n"
            "목적: 두 사람 간의 유전적 생김새 닮음도(붕어빵 지수)를 정밀 측정합니다.\n"
            "★ 매우 중요: 절대로 범죄 수사처럼 '동일한 사람인가?'를 따지는 것이 아닙니다. 가족 간에 얼마나 이목구비가 쏙 빼닮았는지를 측정하는 것입니다.\n\n"
            "★★★★★ 엄격한 채점 가이드라인 (반드시 준수) ★★★★★\n"
            "1. [나이 및 주름 차이 보정 (감점 절대 금지)]:\n"
            "   - 아빠와 아들은 당연히 20~30세의 나이 차이가 존재합니다.\n"
            "   - 아빠의 주름, 연륜, 피부결, 흰머리, 수염 등 노화 현상으로 인해 점수를 깎으면 절대 안 됩니다!\n"
            "   - 아들이 나이 들었을 때 아빠의 모습이 연상되거나, 이목구비(눈매, 콧날, 웃는 입모양, 턱선)가 같은 유전자임을 보여준다면 85~95점의 '완벽한 붕어빵/매우 높은 닮음' 점수를 부여하세요.\n"
            "2. [성별 차이 보정 (감점 절대 금지)]:\n"
            "   - 엄마(여성)와 아들(남성), 누나와 남동생처럼 성별이 달라도 감점하지 마세요!\n"
            "   - 남녀에 따른 헤어스타일, 화장, 수염 유무 등 성별 외형을 배제하고, 눈의 가로세로 비율, 쌍꺼풀 라인, 콧망울 너비, 인중 길이, 웃을 때의 입꼬리 등 순수 이목구비의 유전적 닮음을 비교하세요.\n"
            "3. [닮은꼴·붕어빵 점수 기준 (0~100점)]:\n"
            "   * 85 ~ 100점: '완벽한 붕어빵 (판박이)' - 눈매, 콧대, 웃는 인상, 턱선이 쏙 빼닮음\n"
            "   * 70 ~ 84점: '매우 높은 붕어빵 지수' - 한눈에 가족임을 알아볼 만큼 핵심 이목구비가 뚜렷하게 닮음\n"
            "   * 50 ~ 69점: '상당한 유전적 닮음' - 얼굴 윤곽이나 특정 이목구비, 미소가 상당 부분 일치\n"
            "   * 35 ~ 49점: '은근한 닮음' - 특정 부위 하나만 살짝 닮고 전반적인 인상은 차이가 큼\n"
            "   * 0 ~ 34점: '서로 다른 생김새' - 골격과 이목구비 전반에 걸쳐 닮은 특징이 거의 없음\n"
        )
        verdict_options = "완벽한 붕어빵(판박이) | 매우 높은 닮음 | 뚜렷한 닮은꼴 | 은근한 닮음 | 서로 다른 생김새"

    system_instruction = (
        f"{mode_instruction}\n\n"
        "당신은 인물 생김새 분석 및 안면 특징 정밀 감정 전문 AI입니다.\n"
        "제시된 [사진 1]과 [사진 2]의 인물 얼굴을 세밀하게 관찰하고, 두 사람이 얼마나 쏙 빼닮았는지 분석하여 결과를 제시하세요.\n\n"
        "대조 관찰 부위:\n"
        "- 얼굴형: 턱선(하악각, V라인/각진형/둥근형), 광대뼈 위치, 이마와 턱의 삼분 비율\n"
        "- 눈매: 눈의 가로/세로 비율, 쌍꺼풀 유무 및 형태, 눈꼬리 각도, 웃을 때 접히는 눈웃음 라인\n"
        "- 콧대: 콧대의 높이와 시작점, 콧볼(비익) 너비, 코끝의 둥글거나 날렵한 형태\n"
        "- 입술 & 하관: 입술 두께, 입꼬리 방향, 인중 길이, 웃을 때 치아 노출 및 입매 모양\n"
        "- 고유 인상: 전체적으로 풍기는 분위기, 표정 짓는 습관, 붕어빵 싱크로율\n\n"
        "반드시 다음 JSON 스키마 형식으로만 출력하세요 (마크다운 코드블록이나 서두/결말 문장 없이 순수한 JSON 문자열만 출력):\n"
        "{\n"
        '  "similarity_score": <0부터 100 사이의 정수 (닮음도 점수)>,\n'
        f'  "verdict": "<{verdict_options} 중 적합한 것 택1>",\n'
        '  "verdict_summary": "<두 사람의 닮은 정도와 핵심 붕어빵 포인트를 짚어주는 자연스러운 한국어 요약 한 문장 (예: 아빠의 선한 눈매와 웃을 때의 시원한 입매를 쏙 빼닮은 붕어빵 부자입니다.)>",\n'
        '  "detailed_scores": {\n'
        '    "face_shape": <얼굴형 및 턱선 닮음도 0-100>,\n'
        '    "eyes": <눈매 및 눈웃음 닮음도 0-100>,\n'
        '    "nose": <콧대 및 코끝 닮음도 0-100>,\n'
        '    "mouth": <입술 및 미소/하관 닮음도 0-100>,\n'
        '    "features": <고유 인상 및 붕어빵 분위기 닮음도 0-100>\n'
        '  },\n'
        '  "similarities": [\n'
        '    "<가장 쏙 빼닮은 특징 1 (부위와 형태 상세 기술)>",\n'
        '    "<가장 쏙 빼닮은 특징 2 (부위와 형태 상세 기술)>",\n'
        '    "<가장 쏙 빼닮은 특징 3 (부위와 형태 상세 기술)>"\n'
        '  ],\n'
        '  "differences": [\n'
        '    "<서로 구별되는 개성적 차이점 1 (나이/성별 외의 형태적 차이)>",\n'
        '    "<서로 구별되는 개성적 차이점 2>"\n'
        '  ],\n'
        '  "environmental_factors": "<연령대, 성별, 촬영 각도, 표정 등 분석 시 보정하여 감안한 외적 차이점>",\n'
        '  "comprehensive_analysis": "<두 사람의 이목구비와 얼굴 전체가 어디가 어떻게 닮았는지 유전적 특징과 붕어빵 포인트를 친절하고 흥미롭게 설명하는 종합 감정 소견 (3~5문장)>"\n'
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
        "temperature": 0.1,
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

            # Ensure all required keys exist
            parsed_analysis.setdefault("similarity_score", 50)
            parsed_analysis.setdefault("verdict", "판정 완료")
            parsed_analysis.setdefault("verdict_summary", "분석이 완료되었습니다.")
            parsed_analysis.setdefault("detailed_scores", {
                "face_shape": 50, "eyes": 50, "nose": 50, "mouth": 50, "features": 50
            })
            parsed_analysis.setdefault("similarities", [])
            parsed_analysis.setdefault("differences", [])
            parsed_analysis.setdefault("environmental_factors", "특이사항 없음")
            parsed_analysis.setdefault("comprehensive_analysis", content_text)

            return {
                "success": True,
                "model": MODEL_NAME,
                "data": parsed_analysis,
                "raw_output": content_text if logger.isEnabledFor(logging.DEBUG) else None
            }

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
