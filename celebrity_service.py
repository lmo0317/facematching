"""
Celebrity Lookalike Service (닮은 연예인 탐색 서비스)
1. ArcFace(512-D) 얼굴 임베딩으로 celebrity_db.json 전체에서 가장 가까운 연예인 TOP 3를 선정 (후보 선정·점수는 임베딩만 사용)
2. 성별 필터는 코드에서 강제 (auto: 가장 가까운 연예인들의 성별 다수결로 추정)
3. Gemma 4 E4B는 사용자 사진 + 선정된 연예인 사진을 보고 얼굴 특징과 닮은 이유 설명만 작성
"""

import os
import json
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import httpx

from face_utils import (
    EMBEDDING_DIM,
    image_bytes_to_data_url,
    extract_json_block,
    extract_face_embedding,
    data_url_to_bytes,
    similarity_to_percent,
    adjust_part_scores,
)

logger = logging.getLogger("facematch.celebrity")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")

TOP_K = 5
GEMMA_DESCRIBED = 3        # Gemma explains only the top matches (each extra image adds latency)
SAME_PERSON_MIN_COS = 0.25  # extra photos below this vs the main photo are treated as someone else
GENDER_VOTE_K = 15
PART_KEYS = ["eyes", "nose", "mouth", "face_shape", "features"]

_CACHED_DB: Optional[Tuple[List[Dict[str, Any]], np.ndarray]] = None


class FaceNotFoundError(Exception):
    """No usable face in the user's photo."""


def load_celebrity_db() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """(entries, L2-normalized embedding matrix). Entries whose embedding size mismatches the model are dropped."""
    global _CACHED_DB
    if _CACHED_DB is not None:
        return _CACHED_DB

    entries: List[Dict[str, Any]] = []
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read celebrity_db.json: {e}")

    valid = [e for e in entries if len(e.get("embedding", [])) == EMBEDDING_DIM]
    if len(valid) < len(entries):
        logger.error(f"{len(entries) - len(valid)} DB entries have a non-{EMBEDDING_DIM}-D embedding; "
                     "rebuild with populate_celebrities.py")
    matrix = np.array([e["embedding"] for e in valid], dtype=np.float32).reshape(-1, EMBEDDING_DIM)
    if len(valid):
        _CACHED_DB = (valid, matrix)
    return valid, matrix


def infer_gender(sims: np.ndarray, entries: List[Dict[str, Any]]) -> str:
    """Similarity-weighted vote of the nearest celebrities' genders (ArcFace space separates gender well)."""
    nearest = np.argsort(-sims)[:GENDER_VOTE_K]
    votes = {"male": 0.0, "female": 0.0}
    for i in nearest:
        g = entries[i].get("gender")
        if g in votes:
            votes[g] += max(float(sims[i]), 0.0) + 1e-3
    return max(votes, key=votes.get)


def _photo_data_url(entry: Dict[str, Any]) -> Optional[str]:
    path = os.path.join(CELEB_DIR, entry["filename"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")


async def request_gemma_descriptions(
    user_image_url: str,
    matches: List[Dict[str, Any]],
    llama_url: str,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Ask Gemma to describe the user's face and explain each (already chosen) match. None on failure."""
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": "[사진 0] 사용자:"},
        {"type": "image_url", "image_url": {"url": user_image_url}},
    ]
    for i, m in enumerate(matches, start=1):
        photo = _photo_data_url(m["entry"])
        if photo:
            content.append({"type": "text", "text": f"\n[사진 {i}] {m['entry']['name']} ({m['entry'].get('category', '연예인')}):"})
            content.append({"type": "image_url", "image_url": {"url": photo}})

    names = ", ".join(f"{i}위 {m['entry']['name']}" for i, m in enumerate(matches, start=1))
    prompt = (
        "\n\n당신은 안면 형태학 전문가입니다. 얼굴 인식 AI가 [사진 0] 사용자와 가장 닮은 연예인으로 "
        f"{names}을(를) 이미 선정했습니다.\n"
        "연예인 선정은 끝났으므로 이름을 바꾸거나 다른 연예인을 추천하지 마세요.\n"
        "사진을 직접 비교해서 실제로 보이는 공통점만 구체적으로 쓰세요. 헤어스타일·옷·안경·표정 같은 외적 요소가 아니라 "
        "눈매, 코, 입매, 얼굴형, 이목구비 배치를 중심으로 설명합니다.\n"
        "detailed_scores는 각 부위가 얼마나 닮았는지 0~100 점수입니다. 부위 간 차이가 드러나도록 솔직하게 매기세요.\n\n"
        "반드시 다음 JSON 형식으로만 출력하세요:\n"
        "{\n"
        '  "gender": "<남성 또는 여성>",\n'
        '  "age_group": "<추정 연령대>",\n'
        '  "face_shape": "<사용자 얼굴형 특징 한 줄>",\n'
        '  "eyes": "<사용자 눈매 특징 한 줄>",\n'
        '  "nose_mouth": "<사용자 코와 입매 특징 한 줄>",\n'
        '  "animal_vibe": "<동물상 및 인상 키워드>",\n'
        '  "overall_vibe": "<전체 인상 요약 1~2문장>",\n'
        '  "matches": [\n'
        "    {\n"
        '      "summary": "<이 연예인과 가장 닮은 포인트 한 줄>",\n'
        '      "reason": "<어느 부위가 어떻게 닮았는지 2문장>",\n'
        '      "matching_points": ["<닮은 점 1>", "<닮은 점 2>", "<닮은 점 3>"],\n'
        '      "detailed_scores": {"eyes": 0, "nose": 0, "mouth": 0, "face_shape": 0, "features": 0}\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"matches 배열은 {len(matches)}개이며 1위부터 순서대로 작성하세요."
    )
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": 1400,
        "stream": False,
        "cache_prompt": False,
    }
    try:
        async with httpx.AsyncClient(timeout=75.0) as client:
            resp = await client.post(f"{llama_url}/v1/chat/completions", json=payload,
                                     headers={"Content-Type": "application/json"})
        if resp.status_code != 200:
            logger.warning(f"Gemma description call returned {resp.status_code}")
            return None
        return extract_json_block(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning(f"Gemma description call failed: {e}")
        return None


async def execute_celebrity_lookalike(
    image_b64: str,
    gender_filter: str = "auto",
    llama_url: str = "http://127.0.0.1:8081",
    model_name: str = "gemma-4-e4b-it-q4km",
    extra_images_b64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    raw_bytes = data_url_to_bytes(image_b64)

    entries, matrix = load_celebrity_db()
    if not entries:
        raise RuntimeError("연예인 데이터베이스가 비어 있습니다. populate_celebrities.py로 DB를 생성하세요.")

    main_feat = extract_face_embedding(raw_bytes)
    if main_feat is None:
        raise FaceNotFoundError("사진에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 정면 사진을 올려 주세요.")

    # Averaging several photos of the same person cancels out one photo's angle/lighting/expression
    # (look-alike eval: median rank of the known look-alike 36 -> 18 with 3 photos)
    feats, rejected = [main_feat], 0
    for extra in extra_images_b64 or []:
        try:
            feat = extract_face_embedding(data_url_to_bytes(extra))
        except Exception:
            feat = None
        if feat is not None and float(feat @ main_feat) >= SAME_PERSON_MIN_COS:
            feats.append(feat)
        else:
            rejected += 1
    user_feat = np.mean(feats, axis=0)
    user_feat /= np.linalg.norm(user_feat)

    # 1. Rank every celebrity by ArcFace cosine similarity
    sims = matrix @ user_feat.astype(np.float32)
    gender = gender_filter if gender_filter in ("male", "female") else infer_gender(sims, entries)
    pool = [i for i, e in enumerate(entries) if e.get("gender") == gender] or list(range(len(entries)))
    top = sorted(pool, key=lambda i: -sims[i])[:TOP_K]
    matches = [{"entry": entries[i], "cosine": float(sims[i]),
                "percent": similarity_to_percent(float(sims[i]), "celebrity")} for i in top]

    # 2. Gemma writes the explanations for the fixed matches
    user_url = image_bytes_to_data_url(raw_bytes, max_dim=768)
    gemma = await request_gemma_descriptions(user_url, matches[:GEMMA_DESCRIBED], llama_url, model_name) or {}
    gemma_matches = gemma.get("matches") if isinstance(gemma.get("matches"), list) else []

    formatted = []
    for rank, m in enumerate(matches, start=1):
        e = m["entry"]
        g = gemma_matches[rank - 1] if rank - 1 < len(gemma_matches) and isinstance(gemma_matches[rank - 1], dict) else {}
        name = e["name"]
        formatted.append({
            "rank": rank,
            "name": name,
            "category": e.get("category", "연예인"),
            "similarity_percent": m["percent"],
            "cosine": round(m["cosine"], 4),
            "detailed_scores": adjust_part_scores(g.get("detailed_scores"), m["percent"], PART_KEYS),
            "summary": g.get("summary") or e.get("face_type") or "얼굴 인식 AI가 측정한 눈·코·입 배치와 얼굴 윤곽이 가까운 후보입니다.",
            "reason": g.get("reason") or e.get("vibe") or "눈매와 얼굴형 등 이목구비의 전체적인 배치가 유사합니다.",
            "matching_points": g.get("matching_points") or [p for p in [e.get("face_type"), "이목구비 배치", "얼굴 윤곽"] if p],
            "photo_url": e["photo_url"],
            "photo_source": e.get("photo_source"),
        })

    return {
        "success": True,
        "method": "arcface",
        "gender_used": gender,
        "photos_used": len(feats),
        "photos_rejected": rejected,
        "face_features": {
            "face_type": gemma.get("animal_vibe") or "분석 정보 없음",
            "face_shape": gemma.get("face_shape") or "분석 정보 없음",
            "eyes": gemma.get("eyes") or "분석 정보 없음",
            "nose_mouth": gemma.get("nose_mouth") or "분석 정보 없음",
            "overall_vibe": gemma.get("overall_vibe") or "AI 설명을 생성하지 못해 얼굴 인식 결과만 표시합니다.",
        },
        "top_celebrity": formatted[0] if formatted else None,
        "candidates": formatted[1:],
        "all_celebrities": formatted,
    }
