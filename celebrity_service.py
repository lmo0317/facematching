"""
Celebrity Lookalike Service (정밀 유사 연예인 탐색 서비스)
- Gemma 4 E4B Multimodal 안면 형태학 정밀 관찰 (연령대, 얼굴형, 눈매, 동물상, 분위기 분석)
- 인간 인상학 기반 실물 일치 연예인 매칭 + 위키피디아 실물 사진 자동 연동
- OpenCV SFace(128-D 임베딩) 기하학 정량 검증 결합
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
import httpx

logger = logging.getLogger("facematch.celebrity")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")

YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

os.makedirs(CELEB_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "FaceMatchApp/2.0 (https://minohlee.mooo.com; admin@minohlee.mooo.com)"
}

_CACHED_DB: Optional[List[Dict[str, Any]]] = None
_SFACE_RECOGNIZER = None


def get_sface_recognizer():
    global _SFACE_RECOGNIZER
    if _SFACE_RECOGNIZER is None and os.path.exists(SFACE_PATH):
        try:
            _SFACE_RECOGNIZER = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
        except Exception as e:
            logger.warning(f"Failed to create SFace recognizer: {e}")
    return _SFACE_RECOGNIZER


def load_celebrity_db() -> List[Dict[str, Any]]:
    global _CACHED_DB
    if _CACHED_DB is not None and len(_CACHED_DB) > 0:
        return _CACHED_DB

    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                _CACHED_DB = json.load(f)
            return _CACHED_DB
        except Exception as e:
            logger.error(f"Failed to read celebrity_db.json: {e}")

    return []


def process_image_to_base64(image_bytes: bytes, max_dim: int = 768) -> str:
    """Validate, orient and resize image, then convert to base64 data URL."""
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


def extract_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Extract 128-D normalized SFace feature vector from primary face."""
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

        best_face = max(faces, key=lambda f: f[2] * f[3])
        aligned = recognizer.alignCrop(img_bgr, best_face)
        feat = recognizer.feature(aligned)
        norm = np.linalg.norm(feat)
        if norm > 0:
            return (feat / norm).flatten()
        return None
    except Exception as e:
        logger.warning(f"Error extracting face embedding: {e}")
        return None


def normalize_celeb_name(name: str) -> str:
    cleaned = re.sub(r"\(.*?\)", "", name)
    cleaned = re.sub(r"^(배우|가수|아이돌|개그맨|방송인|모델|선수)\s*", "", cleaned)
    cleaned = re.sub(r"\s*(배우|가수|아이돌|개그맨|방송인|모델|선수)$", "", cleaned)
    return cleaned.strip()


def resolve_celebrity_image(name: str) -> Optional[Tuple[str, str]]:
    """
    Get (web_url, local_path) for celebrity image.
    1. Checks celebrity_db.json
    2. Checks existing local file matching name
    3. Searches Wikipedia/Wikimedia on-demand, downloads, center-crops to square, caches, and returns.
    """
    norm = normalize_celeb_name(name)
    db = load_celebrity_db()

    # 1. Search cached DB
    for item in db:
        if item["name"] == norm or norm in item["name"]:
            local_path = os.path.join(CELEB_DIR, item["filename"])
            if os.path.exists(local_path) and os.path.getsize(local_path) > 4000:
                return item["photo_url"], local_path

    # 2. Search local files
    safe_slug = re.sub(r"[^a-zA-Z0-9가-힣_]", "_", norm) + ".jpg"
    local_path = os.path.join(CELEB_DIR, safe_slug)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 4000:
        return f"/facematching/static/celebrities/{safe_slug}", local_path

    # 3. Search Wikipedia (Korean then English)
    for lang, q in [("ko", norm), ("en", norm)]:
        try:
            encoded = urllib.parse.quote(q)
            url = f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}&gsrlimit=1&prop=pageimages&piprop=original&format=json"
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=6) as r:
                data = json.loads(r.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                img_url = None
                for pid, info in pages.items():
                    if "original" in info and "source" in info["original"]:
                        img_url = info["original"]["source"]
                        break

                if img_url:
                    img_req = urllib.request.Request(img_url, headers=HEADERS)
                    with urllib.request.urlopen(img_req, timeout=10) as ir:
                        raw_bytes = ir.read()

                    with Image.open(io.BytesIO(raw_bytes)) as img:
                        img = img.convert("RGB")
                        w, h = img.size
                        min_dim = min(w, h)
                        left = (w - min_dim) // 2
                        top = max(0, (h - min_dim) // 3)
                        right = left + min_dim
                        bottom = top + min_dim
                        if bottom > h:
                            top = h - min_dim
                            bottom = h
                        cropped = img.crop((left, top, right, bottom)).resize((600, 600), Image.Resampling.LANCZOS)
                        cropped.save(local_path, format="JPEG", quality=85, optimize=True)

                    logger.info(f"Downloaded on-demand celebrity photo for {norm} -> {safe_slug}")
                    return f"/facematching/static/celebrities/{safe_slug}", local_path
        except Exception as e:
            logger.warning(f"On-demand image search failed for {norm} ({lang}): {e}")

    return None


def extract_json_safe(content: str) -> Optional[Dict[str, Any]]:
    """Clean LLM output and parse JSON."""
    cleaned = re.sub(r"<channel>thought.*?</channel>", "", content, flags=re.DOTALL)
    cleaned = re.sub(r"<\|think\|>.*?</turn>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

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


async def execute_celebrity_lookalike(
    image_b64: str,
    gender_filter: str = "auto",
    llama_url: str = "http://127.0.0.1:8081",
    model_name: str = "gemma-4-e4b-it-q4km"
) -> Dict[str, Any]:
    """
    Vision-Guided Archetype & Grounded Photo Matching Engine:
    1. Gemma 4 Vision analyzes user face structure (age group, eyes, jawline, animal vibe).
    2. Gemma selects real lookalike Korean celebrities matching this exact facial archetype.
    3. Resolves high-res photos for top candidates and computes SFace alignment.
    4. Produces grounded, detailed feature breakdown and realistic scores.
    """
    # 1. Ensure clean, properly scaled base64
    data_str = image_b64
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_str)
    processed_b64 = process_image_to_base64(raw_bytes, max_dim=768)

    # 2. Construct Gender Constraint
    gender_instruction = ""
    if gender_filter == "male":
        gender_instruction = "【성별 필터 필수】: 사용자가 '남성 연예인' 매칭을 요청했습니다. 1위부터 5위까지 반드시 한국 '남성' 연예인(배우, 가수, 방송인 등) 중에서만 선정하세요.\n"
    elif gender_filter == "female":
        gender_instruction = "【성별 필터 필수】: 사용자가 '여성 연예인' 매칭을 요청했습니다. 1위부터 5위까지 반드시 한국 '여성' 연예인(배우, 가수, 방송인 등) 중에서만 선정하세요.\n"
    else:
        gender_instruction = "【성별 필터】: 사용자의 실제 성별과 연령대를 관찰하여 가장 자연스러운 한국 연예인 중에서 1위부터 5위까지 선정하세요.\n"

    # 3. Vision Archetype Prompt
    archetype_prompt = (
        "당신은 안면 형태학 및 한국 연예인 인상학 최고 전문가입니다.\n"
        "제시된 [사진 1] 속 인물의 실제 얼굴을 면밀히 관찰하고, 가장 닮은 한국 연예인을 정밀 감정해 주세요.\n\n"
        f"{gender_instruction}\n"
        "【분석 및 채점 원칙】:\n"
        "1. [안면 형태학적 분석]:\n"
        "   - 성별 및 추정 연령대 (예: 20대 후반, 40대 중년 등)\n"
        "   - 얼굴형 (계란형, 둥근형, 사각형, 각진형, 긴형 등)\n"
        "   - 눈매 (무쌍/속쌍/겉쌍, 눈꼬리 방향, 눈매의 느낌)\n"
        "   - 코와 입매 (콧대 높이, 콧망울 너비, 입술 두께와 미소)\n"
        "   - 동물상 / 분위기 (강아지상, 공룡상, 곰상, 두부상, 여우상, 사슴상, 토끼상, 고양이상 등)\n\n"
        "2. [현실적이고 정밀한 닮은꼴 연예인 선정]:\n"
        "   - 단순히 인기 많은 유명 배우를 아무렇게나 찍지 마십시오!\n"
        "   - 실제 사진 속 인물의 '얼굴형, 눈매 형태(쌍꺼풀 유무), 하관 골격, 연령대, 고유 분위기'가 실제로 일치하는 현실적인 한국 연예인 TOP 5를 선정하세요.\n"
        "   - 1위(가장 높은 싱크로율 78~92%), 2위(후보 72~85%), 3위(후보 65~79%)\n"
        "   - 1위 연예인은 눈(eyes), 코(nose), 입(mouth), 얼굴형(face_shape), 분위기(features)의 세부 싱크로율(0~100)을 부여하세요.\n\n"
        "반드시 다음 JSON 형식으로만 순수하게 출력하세요:\n"
        "{\n"
        '  "gender": "<남성 또는 여성>",\n'
        '  "age_group": "<추정 연령대>",\n'
        '  "face_shape": "<얼굴형 상세 설명 한 줄>",\n'
        '  "eyes": "<눈매 특징 상세 설명 한 줄>",\n'
        '  "nose_mouth": "<코와 입매 특징 상세 설명 한 줄>",\n'
        '  "animal_vibe": "<동물상 및 고유 분위기 키워드 (예: 지적인 훈남상, 따뜻한 강아지상)>",\n'
        '  "overall_vibe": "<전체적인 인상과 매력 요약 1~2문장>",\n'
        '  "recommended_celebrities": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "name": "<연예인 실명>",\n'
        '      "category": "<배우 / 가수 / 방송인>",\n'
        '      "similarity_percent": 86,\n'
        '      "detailed_scores": {\n'
        '        "eyes": 88,\n'
        '        "nose": 82,\n'
        '        "mouth": 85,\n'
        '        "face_shape": 86,\n'
        '        "features": 87\n'
        "      },\n"
        '      "summary": "<가장 닮은 핵심 포인트를 친절하게 설명하는 한 줄 요약>",\n'
        '      "reason": "<눈매, 콧날, 입꼬리, 턱선 등 어디가 어떻게 닮았는지 구체적인 이유 2문장>",\n'
        '      "matching_points": ["<닮은 점 1>", "<닮은 점 2>", "<닮은 점 3>"]\n'
        "    },\n"
        "    {\n"
        '      "rank": 2,\n'
        '      "name": "<연예인 실명>",\n'
        '      "category": "<배우 / 가수 / 방송인>",\n'
        '      "similarity_percent": 80,\n'
        '      "summary": "<닮은 점 한 줄 요약>",\n'
        '      "reason": "<닮은 이유 1문장>",\n'
        '      "matching_points": ["<닮은 점 1>", "<닮은 점 2>"]\n'
        "    },\n"
        "    {\n"
        '      "rank": 3,\n'
        '      "name": "<연예인 실명>",\n'
        '      "category": "<배우 / 가수 / 방송인>",\n'
        '      "similarity_percent": 74,\n'
        '      "summary": "<닮은 점 한 줄 요약>",\n'
        '      "reason": "<닮은 이유 1문장>",\n'
        '      "matching_points": ["<닮은 점 1>"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "다음 인물의 실제 사진을 자세히 관찰하고 닮은 연예인을 정밀 감정해 주세요.\n\n[사진 1]:"},
                    {"type": "image_url", "image_url": {"url": processed_b64}},
                    {"type": "text", "text": f"\n\n{archetype_prompt}"}
                ]
            }
        ],
        "temperature": 0.25,
        "max_tokens": 1200,
        "stream": False,
        "cache_prompt": False
    }

    gemma_data = None
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                f"{llama_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if resp.status_code == 200:
                result_json = resp.json()
                content = result_json["choices"][0]["message"]["content"]
                gemma_data = extract_json_safe(content)
    except Exception as e:
        logger.warning(f"Gemma 4 archetype call failed: {e}")

    # Fallback to SFace database ranking if Gemma failed
    if not gemma_data or not gemma_data.get("recommended_celebrities"):
        logger.warning("Gemma archetype failed, falling back to SFace database")
        user_feat = extract_face_embedding(raw_bytes)
        db = load_celebrity_db()
        filtered_db = [c for c in db if gender_filter in ["auto", c.get("gender")]] or db

        top_candidates = []
        if user_feat is not None:
            for item in filtered_db:
                c_feat = np.array(item["embedding"], dtype=np.float32)
                sim = float(np.dot(user_feat, c_feat))
                top_candidates.append((item, sim))
            top_candidates.sort(key=lambda x: x[1], reverse=True)

        if not top_candidates:
            top_candidates = [(c, 0.3) for c in filtered_db[:3]]

        recs = []
        for i, (item, sim) in enumerate(top_candidates[:3], start=1):
            score = max(65, min(95, int(round(65 + sim * 70))))
            recs.append({
                "rank": i,
                "name": item["name"],
                "category": item.get("category", "배우"),
                "similarity_percent": score,
                "detailed_scores": {"eyes": score, "nose": score, "mouth": score, "face_shape": score, "features": score},
                "summary": f"{item['name']} 특유의 {item.get('face_type', '매력적인 인상')}과 싱크로율을 보입니다.",
                "reason": item.get("vibe", "이목구비 비율과 전체적인 분위기가 유사합니다."),
                "matching_points": [item.get("face_type", "분위기 닮음"), "이목구비 밸런스"]
            })

        gemma_data = {
            "face_shape": "균형 잡힌 자연스러운 윤곽선",
            "eyes": "선하고 매력적인 눈매",
            "nose_mouth": "오뚝하고 단정한 콧날과 입매",
            "animal_vibe": "호감형 훈남상",
            "overall_vibe": "전체적으로 단정하고 신뢰감을 주는 매력적인 인상입니다.",
            "recommended_celebrities": recs
        }

    # 4. Resolve real photos and clean scores
    recs = gemma_data.get("recommended_celebrities", [])
    formatted_celebs = []

    user_feat = extract_face_embedding(raw_bytes)

    for idx, c in enumerate(recs[:3], start=1):
        c_name = c.get("name", "연예인")
        resolved = resolve_celebrity_image(c_name)
        photo_url = resolved[0] if resolved else f"/facematching/static/celebrities/gong_yoo.jpg"
        local_path = resolved[1] if resolved else None

        # Clean raw similarity score
        raw_score = c.get("similarity_percent", 80)
        try:
            score = max(60, min(96, int(round(float(raw_score)))))
        except (ValueError, TypeError):
            score = 80

        # Fine-tune score with SFace geometric similarity if photo available
        if user_feat is not None and local_path and os.path.exists(local_path):
            try:
                with open(local_path, "rb") as cf:
                    c_feat = extract_face_embedding(cf.read())
                if c_feat is not None:
                    sface_sim = float(np.dot(user_feat, c_feat))
                    # Cosine bonus/penalty: 0.35+ boosts score, 0.15- lowers score
                    sface_bonus = int(round((sface_sim - 0.25) * 25))
                    score = max(62, min(96, score + sface_bonus))
            except Exception:
                pass

        det = c.get("detailed_scores", {})
        cleaned_det = {}
        for part in ["eyes", "nose", "mouth", "face_shape", "features"]:
            raw_p = det.get(part, score)
            try:
                cleaned_det[part] = max(50, min(100, int(round(float(raw_p)))))
            except (ValueError, TypeError):
                cleaned_det[part] = score

        formatted_celebs.append({
            "rank": idx,
            "name": c_name,
            "category": c.get("category", "연예인"),
            "similarity_percent": score,
            "detailed_scores": cleaned_det,
            "summary": c.get("summary", f"{c_name}과(와) 이목구비 밸런스와 분위기가 매우 유사합니다."),
            "reason": c.get("reason", f"얼굴의 전반적인 분위기와 이목구비 골격이 {c_name}과(와) 닮은 느낌을 줍니다."),
            "matching_points": c.get("matching_points", [f"{c_name} 특유의 매력", "자연스러운 눈매", "시원한 입꼬리"]),
            "photo_url": photo_url
        })

    # Sort so rank 1 is highest
    formatted_celebs.sort(key=lambda x: x["similarity_percent"], reverse=True)
    for i, c in enumerate(formatted_celebs, start=1):
        c["rank"] = i

    top_celeb = formatted_celebs[0] if formatted_celebs else None
    candidates = formatted_celebs[1:] if len(formatted_celebs) > 1 else []

    user_face_features = {
        "face_type": gemma_data.get("animal_vibe", "매력적인 호감상"),
        "face_shape": gemma_data.get("face_shape", "균형 잡힌 자연스러운 윤곽선"),
        "eyes": gemma_data.get("eyes", "선하고 매력적인 눈매"),
        "nose_mouth": gemma_data.get("nose_mouth", "오뚝하고 단정한 콧날과 입매"),
        "overall_vibe": gemma_data.get("overall_vibe", "전체적으로 단정하고 신뢰감을 주는 매력적인 인상입니다.")
    }

    return {
        "success": True,
        "face_features": user_face_features,
        "top_celebrity": top_celeb,
        "candidates": candidates,
        "all_celebrities": formatted_celebs
    }
