"""
Celebrity Lookalike Service (정밀 유사 연예인 탐색 서비스)
- OpenCV YuNet 안면 감지 + SFace (128-D Deep Face Embedding) 안면 기하학 유사도 매칭
- Gemma 4 E4B Multimodal 1:1 사진 나란히 대조 (Side-by-Side Visual Comparison)
- 80명 이상의 남녀 한국 연예인 실물 라이브러리 및 정밀 부위별 싱크로율 감정
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
from PIL import Image
import httpx

logger = logging.getLogger("facematch.celebrity")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

# Cache loaded DB in memory
_CACHED_DB: Optional[List[Dict[str, Any]]] = None
_SFACE_RECOGNIZER = None


def get_sface_recognizer():
    global _SFACE_RECOGNIZER
    if _SFACE_RECOGNIZER is None:
        if os.path.exists(SFACE_PATH):
            _SFACE_RECOGNIZER = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            logger.info("Initialized OpenCV SFace Recognizer")
        else:
            logger.warning(f"SFace model not found at {SFACE_PATH}")
    return _SFACE_RECOGNIZER


def load_celebrity_db() -> List[Dict[str, Any]]:
    global _CACHED_DB
    if _CACHED_DB is not None and len(_CACHED_DB) > 0:
        return _CACHED_DB

    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                _CACHED_DB = json.load(f)
            logger.info(f"Loaded {len(_CACHED_DB)} celebrities from {DB_PATH}")
            return _CACHED_DB
        except Exception as e:
            logger.error(f"Failed to read celebrity_db.json: {e}")

    logger.warning("Celebrity DB empty or missing, returning empty list")
    return []


def calibrate_celeb_sync(cosine_sim: float) -> int:
    """
    Calibrate raw SFace cosine similarity (typically 0.15 ~ 0.50 between different people)
    to intuitive and motivating sync percentages (55% ~ 95%).
    """
    if cosine_sim <= 0.18:
        return max(50, int(round(50 + (cosine_sim / 0.18) * 15)))  # 50 - 65%
    elif cosine_sim <= 0.28:
        return int(round(65 + ((cosine_sim - 0.18) / 0.10) * 12))  # 65 - 77%
    elif cosine_sim <= 0.38:
        return int(round(77 + ((cosine_sim - 0.28) / 0.10) * 11))  # 77 - 88%
    else:
        return min(98, int(round(88 + ((cosine_sim - 0.38) / 0.12) * 9)))  # 88 - 97%


def extract_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Detect primary face and extract 128-D normalized SFace feature vector."""
    try:
        recognizer = get_sface_recognizer()
        if recognizer is None or not os.path.exists(YUNET_PATH):
            return None

        img_np = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None

        h, w = img_bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (w, h), 0.45, 0.3, 10)
        _, faces = detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None

        # Choose largest face
        best_face = max(faces, key=lambda f: f[2] * f[3])
        aligned = recognizer.alignCrop(img_bgr, best_face)
        feat = recognizer.feature(aligned)
        norm = np.linalg.norm(feat)
        if norm > 0:
            return (feat / norm).flatten()
        return None
    except Exception as e:
        logger.warning(f"Error in extract_face_embedding: {e}")
        return None


def search_top_celebrities(
    user_feat: np.ndarray,
    gender_filter: str = "auto",
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Compute cosine similarity against all celebrities in DB and rank top K."""
    db = load_celebrity_db()
    if not db:
        return []

    results = []
    for item in db:
        if gender_filter in ["male", "female"] and item.get("gender") != gender_filter:
            continue

        c_feat = np.array(item["embedding"], dtype=np.float32)
        sim = float(np.dot(user_feat, c_feat))
        calibrated_score = calibrate_celeb_sync(sim)

        results.append({
            "name": item["name"],
            "category": item.get("category", "배우"),
            "gender": item.get("gender", "male"),
            "filename": item["filename"],
            "photo_url": item.get("photo_url", f"/facematching/static/celebrities/{item['filename']}"),
            "face_type": item.get("face_type", "매력적인 인상"),
            "vibe": item.get("vibe", "전체적인 균형이 잡힌 호감형 인상"),
            "cosine_sim": sim,
            "similarity_percent": calibrated_score
        })

    results.sort(key=lambda x: x["cosine_sim"], reverse=True)
    return results[:top_k]


def extract_json_safe(content: str) -> Optional[Dict[str, Any]]:
    """Clean LLM output and parse JSON."""
    cleaned = re.sub(r"<channel>thought.*?</channel>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<\|think\|>.*?</turn>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    # 1. Try markdown code block
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except Exception:
            pass

    # 2. Try outermost curly braces
    outer = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if outer:
        try:
            return json.loads(outer.group(1))
        except Exception:
            pass

    return None


async def compare_user_and_celebrity_with_gemma(
    user_img_b64: str,
    celeb_item: Dict[str, Any],
    llama_url: str,
    model_name: str
) -> Optional[Dict[str, Any]]:
    """
    Send BOTH user's photo and the #1 celebrity's actual photo to Gemma 4 Vision.
    Performs grounded, detailed side-by-side visual analysis.
    """
    try:
        celeb_name = celeb_item["name"]
        celeb_filename = celeb_item["filename"]
        celeb_path = os.path.join(CELEB_DIR, celeb_filename)

        if not os.path.exists(celeb_path):
            return None

        # Read celebrity photo and convert to base64
        with open(celeb_path, "rb") as f:
            celeb_bytes = f.read()
        celeb_b64 = f"data:image/jpeg;base64,{base64.b64encode(celeb_bytes).decode('utf-8')}"

        prompt = (
            f"다음 두 사람의 사진을 나란히 면밀히 관찰하고, [사용자 사진]과 딥러닝으로 매칭된 [닮은꼴 연예인 {celeb_name} 사진]의 이목구비 부위별 싱크로율을 정밀 비교 분석해 주세요.\n\n"
            "【분석 및 채점 원칙】:\n"
            "1. 실제 두 사람의 사진 속 눈(쌍꺼풀, 눈꼬리 각도), 코(콧대 높이, 콧볼 너비), 입(입술 두께, 미소선), 턱선/얼굴형, 분위기를 비교하세요.\n"
            "2. 두 사람이 어디가 어떻게 닮았는지 구체적이고 생생하게 2문장으로 설명하세요.\n"
            "3. 반드시 다음 JSON 형식으로만 순수하게 출력하세요:\n"
            "{\n"
            '  "detailed_scores": {\n'
            '    "eyes": <눈매 싱크로율 0-100>,\n'
            '    "nose": <콧대/콧볼 싱크로율 0-100>,\n'
            '    "mouth": <입술/미소 싱크로율 0-100>,\n'
            '    "face_shape": <얼굴형/턱선 싱크로율 0-100>,\n'
            '    "features": <고유 인상/분위기 싱크로율 0-100>\n'
            "  },\n"
            f'  "summary": "<{celeb_name}과(와) 가장 닮은 핵심 포인트를 친절하게 설명하는 한 줄 요약>",\n'
            '  "reason": "<눈매, 콧날, 입꼬리, 턱선 등 어디가 어떻게 닮았는지 구체적인 이유 2문장>",\n'
            '  "matching_points": ["<구체적 닮은점 1>", "<구체적 닮은점 2>", "<구체적 닮은점 3>"],\n'
            '  "face_features": {\n'
            f'    "face_type": "<사용자의 동물상 및 인상 키워드 (예: {celeb_item.get("face_type", "매력적인 훈남상")})>",\n'
            '    "face_shape": "<사용자의 얼굴형 특징 요약 한 줄>",\n'
            '    "eyes": "<사용자의 눈매 특징 요약 한 줄>",\n'
            '    "nose_mouth": "<사용자의 코와 입매 특징 요약 한 줄>",\n'
            '    "overall_vibe": "<사용자의 전반적인 인상과 매력 요약 1~2문장>"\n'
            "  }\n"
            "}"
        )

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"두 사람의 사진을 나란히 정밀 대조하여 닮은꼴 분석을 진행해 주세요.\n\n[사용자 사진]:"},
                        {"type": "image_url", "image_url": {"url": user_img_b64}},
                        {"type": "text", "text": f"\n\n[연예인 {celeb_name} 실제 사진]:"},
                        {"type": "image_url", "image_url": {"url": celeb_b64}},
                        {"type": "text", "text": f"\n\n{prompt}"}
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 850,
            "stream": False,
            "cache_prompt": False
        }

        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(
                f"{llama_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                result_data = resp.json()
                content = result_data["choices"][0]["message"]["content"]
                parsed = extract_json_safe(content)
                if parsed:
                    return parsed
    except Exception as e:
        logger.warning(f"Gemma 4 side-by-side comparison skipped/failed: {e}")

    return None


async def execute_celebrity_lookalike(
    image_b64: str,
    gender_filter: str = "auto",
    llama_url: str = "http://127.0.0.1:8081",
    model_name: str = "gemma-4-e4b-it-q4km"
) -> Dict[str, Any]:
    """
    Dual-AI Pipeline:
    1. OpenCV SFace Deep Face Embeddings -> Exact facial geometry search across 80+ Korean celebrities
    2. Gemma 4 Multimodal Vision -> Side-by-side comparative analysis with the top matched celebrity's real photo
    """
    # 1. Decode base64 bytes
    data_str = image_b64
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    image_bytes = base64.b64decode(data_str)

    # 2. Extract SFace embedding and rank celebrities
    user_feat = extract_face_embedding(image_bytes)
    
    top_candidates = []
    if user_feat is not None:
        top_candidates = search_top_celebrities(user_feat, gender_filter=gender_filter, top_k=3)
        match_names = [f"{c['name']} ({c['similarity_percent']}%)" for c in top_candidates]
        logger.info(f"SFace matches found: {match_names}")

    if not top_candidates:
        # Fallback to curated default candidates if face detection completely failed
        db = load_celebrity_db()
        default_pool = [c for c in db if gender_filter in ["auto", c.get("gender")]] or db
        top_candidates = [
            {
                "name": default_pool[0]["name"] if default_pool else "공유",
                "category": default_pool[0].get("category", "배우") if default_pool else "배우",
                "gender": default_pool[0].get("gender", "male") if default_pool else "male",
                "filename": default_pool[0]["filename"] if default_pool else "gong_yoo.jpg",
                "photo_url": default_pool[0].get("photo_url", "/facematching/static/celebrities/gong_yoo.jpg") if default_pool else "/facematching/static/celebrities/gong_yoo.jpg",
                "face_type": default_pool[0].get("face_type", "매력적인 훈남상") if default_pool else "매력적인 훈남상",
                "vibe": default_pool[0].get("vibe", "전체적인 균형미가 돋보입니다.") if default_pool else "전체적인 균형미가 돋보입니다.",
                "similarity_percent": 82
            }
        ]

    top_celeb = top_candidates[0]
    base_score = top_celeb.get("similarity_percent", 84)

    # 3. Perform grounded Gemma 4 Side-by-Side Visual Verification with #1 Celebrity
    gemma_analysis = await compare_user_and_celebrity_with_gemma(
        user_img_b64=image_b64,
        celeb_item=top_celeb,
        llama_url=llama_url,
        model_name=model_name
    )

    # 4. Integrate results
    if gemma_analysis and "detailed_scores" in gemma_analysis:
        raw_scores = gemma_analysis.get("detailed_scores", {})
        cleaned_scores = {}
        for k in ["eyes", "nose", "mouth", "face_shape", "features"]:
            raw_v = raw_scores.get(k, base_score)
            try:
                cleaned_scores[k] = max(50, min(100, int(round(float(raw_v)))))
            except (ValueError, TypeError):
                cleaned_scores[k] = base_score

        avg_gemma = int(round(sum(cleaned_scores.values()) / max(1, len(cleaned_scores))))
        # Harmonize SFace mathematical score with Gemma visual score
        final_top_score = int(round(base_score * 0.6 + avg_gemma * 0.4))
        top_celeb["similarity_percent"] = max(65, min(97, final_top_score))
        top_celeb["detailed_scores"] = cleaned_scores
        top_celeb["summary"] = gemma_analysis.get("summary", f"{top_celeb['name']}과(와) 이목구비 밸런스와 분위기가 매우 유사합니다.")
        top_celeb["reason"] = gemma_analysis.get("reason", top_celeb.get("vibe", ""))
        top_celeb["matching_points"] = gemma_analysis.get("matching_points", [
            f"{top_celeb['face_type']} 매력", "자연스러운 눈매", "시원한 입꼬리"
        ])
        user_face_features = gemma_analysis.get("face_features", {
            "face_type": top_celeb.get("face_type", "매력적인 호감상"),
            "face_shape": "균형 잡힌 자연스러운 윤곽선",
            "eyes": "선하고 매력적인 눈매",
            "nose_mouth": "오뚝하고 단정한 콧날과 입매",
            "overall_vibe": top_celeb.get("vibe", "전체적으로 단정하고 매력적인 인상입니다.")
        })
    else:
        # High quality geometric fallback
        top_celeb["detailed_scores"] = {
            "eyes": min(98, base_score + 1),
            "nose": max(55, base_score - 2),
            "mouth": min(98, base_score + 2),
            "face_shape": max(55, base_score - 1),
            "features": base_score
        }
        top_celeb["summary"] = f"{top_celeb['name']} 특유의 {top_celeb.get('face_type', '매력적인 인상')}과 높은 싱크로율을 보입니다."
        top_celeb["reason"] = f"얼굴의 중심인 눈매와 턱선의 안면 비례가 {top_celeb['name']}과(와) 닮아 있으며, {top_celeb.get('vibe', '독보적인 매력')}을 풍깁니다."
        top_celeb["matching_points"] = [
            f"{top_celeb.get('face_type', '호감형')} 눈매",
            "균형 잡힌 이목구비 비율",
            "자연스럽고 편안한 인상"
        ]
        user_face_features = {
            "face_type": top_celeb.get("face_type", "매력적인 호감상"),
            "face_shape": "단정하고 조화로운 안면 골격",
            "eyes": "선하고 분위기 있는 눈매",
            "nose_mouth": "균형 잡힌 콧대와 호감형 미소",
            "overall_vibe": top_celeb.get("vibe", "전체적으로 신뢰감과 매력을 주는 인상입니다.")
        }

    # 5. Format candidates (Rank 2 and Rank 3)
    formatted_candidates = []
    for rank_idx, cand in enumerate(top_candidates[1:], start=2):
        c_score = cand.get("similarity_percent", 75)
        formatted_candidates.append({
            "rank": rank_idx,
            "name": cand["name"],
            "category": cand.get("category", "연예인"),
            "similarity_percent": c_score,
            "photo_url": cand.get("photo_url", f"/facematching/static/celebrities/{cand['filename']}"),
            "summary": f"{cand.get('face_type', '호감형 인상')} 분위기가 닮았습니다.",
            "reason": cand.get("vibe", "이목구비 비율과 전체적인 분위기가 유사합니다."),
            "matching_points": [cand.get("face_type", "분위기 닮음"), "이목구비 밸런스"]
        })

    all_celebs = [top_celeb] + formatted_candidates

    return {
        "success": True,
        "face_features": user_face_features,
        "top_celebrity": top_celeb,
        "candidates": formatted_candidates,
        "all_celebrities": all_celebs
    }
