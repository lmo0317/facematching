import os
import io
import re
import json
import base64
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps
import httpx

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

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
        "similarity_score": 50,
        "verdict": "분석 완료 (비정형 응답)",
        "verdict_summary": "상세 분석 텍스트를 참고해 주세요.",
        "detailed_scores": {
            "face_shape": 50,
            "eyes": 50,
            "nose": 50,
            "mouth": 50,
            "features": 50
        },
        "similarities": ["세부 내용은 종합 분석을 확인하세요."],
        "differences": ["세부 내용은 종합 분석을 확인하세요."],
        "environmental_factors": "조명 및 각도 차이가 존재할 수 있습니다.",
        "comprehensive_analysis": cleaned.strip()
    }


class CompareJsonRequest(BaseModel):
    image1_base64: str
    image2_base64: str
    mode: Optional[str] = "detailed"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>FaceMatch AI - 로딩 중...</h1>")


@app.get("/api/health")
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
async def get_samples():
    """List sample image pairs available for instant testing."""
    sample_pairs = [
        {
            "id": "pair1",
            "title": "예제 1 (동일 인물 - 각도/조명 변화)",
            "description": "동일 인물의 스튜디오 정면 사진 vs 야외 미소 사진",
            "expected": "동일 인물 유력 (85% 이상)",
            "img1": "/static/samples/sample1_a.jpg",
            "img2": "/static/samples/sample1_b.jpg"
        },
        {
            "id": "pair2",
            "title": "예제 2 (다른 인물 - 남성 A vs 남성 B)",
            "description": "서로 다른 두 남성의 얼굴 이목구비 정밀 대조",
            "expected": "다른 인물 (40% 이하)",
            "img1": "/static/samples/sample2_a.jpg",
            "img2": "/static/samples/sample2_b.jpg"
        },
        {
            "id": "pair3",
            "title": "예제 3 (다른 인물 - 안경 착용 남녀 대조)",
            "description": "안경 착용 인물 간 골격 및 이목구비 차이 분석",
            "expected": "다른 인물 (35% 이하)",
            "img1": "/static/samples/sample3_a.jpg",
            "img2": "/static/samples/sample3_b.jpg"
        }
    ]
    return {"samples": sample_pairs}


@app.post("/api/compare")
async def compare_images(
    image1: Optional[UploadFile] = File(None),
    image2: Optional[UploadFile] = File(None),
    mode: Optional[str] = Form("detailed")
):
    """Multipart upload comparison endpoint."""
    if not image1 or not image2:
        raise HTTPException(status_code=400, detail="사진 2장을 모두 업로드해 주세요.")

    bytes1 = await image1.read()
    bytes2 = await image2.read()

    data_url1 = process_image_to_base64(bytes1)
    data_url2 = process_image_to_base64(bytes2)

    return await execute_face_comparison(data_url1, data_url2, mode)


@app.post("/api/compare-json")
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

    return await execute_face_comparison(img1, img2, req.mode or "detailed")


async def execute_face_comparison(img1_url: str, img2_url: str, mode: str) -> Dict[str, Any]:
    """Call Gemma 4 E4B multimodal LLM and perform facial similarity evaluation."""
    system_instruction = (
        "당신은 최고 수준의 디지털 안면 감정 및 생체 유사도 판독 전문 AI입니다.\n"
        "제시된 [사진 1]과 [사진 2]의 인물 얼굴을 정밀하게 대조하여 분석하세요.\n\n"
        "분석 및 판정 가이드라인:\n"
        "1. 성별, 골격 구조, 얼굴형(하악각, 광대, 이마 비율), 눈(쌍꺼풀, 눈꼬리, 내안각), "
        "코(비근부 높이, 비익 너비, 코끝 형태), 입(상하순 두께, 인중 길이, 입꼬리)을 객관적으로 대조하세요.\n"
        "2. 안경, 헤어스타일, 조명 차이, 표정 변화(미소 등)에 현혹되지 말고, 실제 골격과 이목구비의 본질적 일치 여부를 파악하세요.\n"
        "3. 유사도 점수(similarity_score) 기준:\n"
        "   - 80 ~ 100점: 동일 인물 확실 / 유력 (골격 및 이목구비 비율이 극히 일치)\n"
        "   - 50 ~ 79점: 닮은꼴 / 혈연 가능성 (유사한 분위기이나 세부 골격 차이 존재)\n"
        "   - 0 ~ 49점: 서로 다른 인물 (골격 또는 성별이 확연히 다름)\n\n"
        "반드시 다음 JSON 스키마 형식에 맞춰 정확한 JSON 데이터만 출력하세요 (마크다운 코드블록이나 서두/결말 없이 JSON 문자열만 출력):\n"
        "{\n"
        '  "similarity_score": <0부터 100 사이의 정수>,\n'
        '  "verdict": "<동일 인물 확실 | 동일 인물 유력 | 유사한 인물(닮은꼴) | 다른 인물 중 택1>",\n'
        '  "verdict_summary": "<핵심 요약 한 문장 (한국어)>",\n'
        '  "detailed_scores": {\n'
        '    "face_shape": <얼굴형 유사도 0-100 사이 정수>,\n'
        '    "eyes": <눈매 유사도 0-100 사이 정수>,\n'
        '    "nose": <콧대 유사도 0-100 사이 정수>,\n'
        '    "mouth": <입술 유사도 0-100 사이 정수>,\n'
        '    "features": <고유특징 유사도 0-100 사이 정수>\n'
        '  },\n'
        '  "similarities": [\n'
        '    "<실제 관찰된 주요 일치점 1>",\n'
        '    "<실제 관찰된 주요 일치점 2>"\n'
        '  ],\n'
        '  "differences": [\n'
        '    "<실제 관찰된 주요 차이점 1>",\n'
        '    "<실제 관찰된 주요 차이점 2>"\n'
        '  ],\n'
        '  "environmental_factors": "<조명, 촬영 각도, 표정, 배경, 안경 등 외적 환경 요인 분석>",\n'
        '  "comprehensive_analysis": "<두 사진 속 인물에 대한 전문적이고 종합적인 안면 대조 감정 소견 (상세 서술)>"\n'
        "}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "다음 두 장의 사진 속 인물 얼굴을 비교 감정해 주세요.\n\n[사진 1]:"},
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
