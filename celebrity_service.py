"""
Celebrity Lookalike Service (닮은 연예인 탐색 서비스)
1. ArcFace(512-D) 얼굴 임베딩으로 celebrity_db.json 전체에서 가까운 연예인 후보를 선정
2. 거의 동점인 상위 후보는 화면에 보이는 사진끼리의 인상(CLIP) 유사도로 순서를 보정해 TOP 5 확정
3. 성별 필터는 코드에서 강제 (auto: 가장 가까운 연예인들의 성별 다수결로 추정)
4. Gemma 4 E4B는 사용자 사진 + 상위 연예인 사진을 보고 얼굴 특징과 닮은 이유 설명만 작성
"""

import os
import json
import base64
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import httpx

from face_utils import (
    EMBEDDING_DIM,
    image_bytes_to_data_url,
    extract_json_block,
    extract_face_features,
    best_matching_embedding,
    data_url_to_bytes,
    similarity_to_percent,
    celebrity_match_strength,
    adjust_part_scores,
)

logger = logging.getLogger("facematch.celebrity")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")
VISUAL_PATH = os.path.join(BASE_DIR, "celebrity_visual.npz")

TOP_K = 5
GEMMA_DESCRIBED = 5        # Gemma explains every shown match (users asked for detail on all of them)
SAME_PERSON_MIN_COS = 0.25  # extra photos below this vs the main photo are treated as someone else
GENDER_VOTE_K = 15
# The top ArcFace candidates are near-ties (median #1-#2 gap 0.017, far below photo-to-photo noise), and
# ArcFace ignores hair/makeup/expression, so the shown #1 was the visually closest of the top 5 only 23%
# of the time. Within the top VISUAL_POOL_K, add VISUAL_WEIGHT x (z-scored CLIP similarity of the shown
# photos): look-alike eval accuracy unchanged, shown #1 visually closest 23% -> 40%.
VISUAL_POOL_K = 10
VISUAL_WEIGHT = 0.03
PART_KEYS = ["eyes", "nose", "mouth", "face_shape", "features"]

_CACHED_DB: Optional[Tuple[List[Dict[str, Any]], np.ndarray]] = None
_CACHED_VISUAL: Optional[np.ndarray] = None


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


def load_visual_matrix(entries: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """CLIP embeddings of the display photos aligned with entries (zero rows where missing), or None."""
    global _CACHED_VISUAL
    if _CACHED_VISUAL is not None and len(_CACHED_VISUAL) == len(entries):
        return _CACHED_VISUAL
    if not os.path.exists(VISUAL_PATH):
        return None
    try:
        data = np.load(VISUAL_PATH)
        pos = {q: i for i, q in enumerate(data["qids"].tolist())}
        emb = data["emb"].astype(np.float32)
        visual = np.zeros((len(entries), emb.shape[1]), dtype=np.float32)
        for i, e in enumerate(entries):
            if e["qid"] in pos:
                visual[i] = emb[pos[e["qid"]]]
        _CACHED_VISUAL = visual
        return visual
    except Exception as e:
        logger.warning(f"Failed to load visual index: {e}")
        return None


def visual_rerank(sims: np.ndarray, candidates: List[int], user_visual: Optional[np.ndarray],
                  visual: Optional[np.ndarray]) -> Tuple[List[int], Dict[int, float]]:
    """Order candidates by ArcFace similarity, nudging near-ties by how alike the shown photos look."""
    ranked = sorted(candidates, key=lambda i: -sims[i])
    scores = {i: float(sims[i]) for i in ranked}
    if user_visual is None or visual is None:
        return ranked, scores
    pool = ranked[:VISUAL_POOL_K]
    v = visual[pool] @ user_visual
    has = np.linalg.norm(visual[pool], axis=1) > 0
    if has.sum() >= 2:
        z = (v - v[has].mean()) / (v[has].std() + 1e-9)
        for i, zi, ok in zip(pool, z, has):
            if ok:
                scores[i] += VISUAL_WEIGHT * float(zi)
    return sorted(candidates, key=lambda i: -scores[i]), scores


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


DETAILED_MATCH_SCHEMA = (
    "    {\n"
    '      "summary": "<이 연예인과 가장 닮은 포인트 한 문장>",\n'
    '      "reason": "<어느 부위가 어떻게 닮았는지 구체적으로 3문장>",\n'
    '      "part_notes": {"eyes": "<눈매 비교 한 문장>", "nose": "<코 비교 한 문장>", '
    '"mouth": "<입매 비교 한 문장>", "face_shape": "<얼굴형·턱선 비교 한 문장>"},\n'
    '      "differences": ["<눈에 띄게 다른 점 1>", "<다른 점 2>"],\n'
    '      "matching_points": ["<닮은 점 키워드 1>", "<닮은 점 키워드 2>", "<닮은 점 키워드 3>"],\n'
    '      "detailed_scores": {"eyes": 0, "nose": 0, "mouth": 0, "face_shape": 0, "features": 0}\n'
    "    }"
)
BRIEF_MATCH_SCHEMA = (
    "    {\n"
    '      "summary": "<가장 닮은 포인트 한 문장>",\n'
    '      "reason": "<어느 부위가 어떻게 닮았는지 2문장>",\n'
    '      "differences": ["<눈에 띄게 다른 점 한 가지>"],\n'
    '      "matching_points": ["<닮은 점 키워드 1>", "<닮은 점 키워드 2>"],\n'
    '      "detailed_scores": {"eyes": 0, "nose": 0, "mouth": 0, "face_shape": 0, "features": 0}\n'
    "    }"
)
# Compact variants for the interactive "top" call: same sections, fewer sentences.
# Generation runs ~75 tok/s, so halving the output (~670 -> ~350 tokens) saves ~4s.
COMPACT_MATCH_SCHEMA = (
    "    {\n"
    '      "summary": "<가장 닮은 포인트, 40자 이내>",\n'
    '      "reason": "<어느 부위가 어떻게 닮았는지, 80자 이내>",\n'
    '      "part_notes": {"eyes": "<눈매 비교, 30자 이내>", "nose": "<코 비교, 30자 이내>", '
    '"mouth": "<입매 비교, 30자 이내>", "face_shape": "<얼굴형 비교, 30자 이내>"},\n'
    '      "differences": ["<눈에 띄게 다른 점, 40자 이내>"],\n'
    '      "matching_points": ["<닮은 점 키워드>", "<키워드>", "<키워드>"],\n'
    '      "detailed_scores": {"eyes": 0, "nose": 0, "mouth": 0, "face_shape": 0, "features": 0}\n'
    "    }"
)
COMPACT_FEATURES_SCHEMA = (
    '  "gender": "<남성 또는 여성>",\n'
    '  "age_group": "<추정 연령대>",\n'
    '  "impression_keywords": ["<인상 키워드>", "<키워드>", "<키워드>"],\n'
    '  "animal_vibe": "<동물상 한 단어>",\n'
    '  "face_shape": "<얼굴형·턱선 특징, 40자 이내>",\n'
    '  "eyes": "<눈매 특징, 40자 이내>",\n'
    '  "nose": "<코 특징, 40자 이내>",\n'
    '  "mouth": "<입매 특징, 40자 이내>",\n'
    '  "overall_vibe": "<전체 인상과 매력, 70자 이내>",\n'
)
USER_FEATURES_SCHEMA = (
    '  "gender": "<남성 또는 여성>",\n'
    '  "age_group": "<추정 연령대>",\n'
    '  "impression_keywords": ["<인상 키워드 1>", "<인상 키워드 2>", "<인상 키워드 3>"],\n'
    '  "animal_vibe": "<동물상 한 단어와 짧은 설명>",\n'
    '  "face_shape": "<사용자 얼굴형·턱선 특징 2문장>",\n'
    '  "eyes": "<사용자 눈매 특징 2문장>",\n'
    '  "nose": "<사용자 코 특징 1~2문장>",\n'
    '  "mouth": "<사용자 입매 특징 1~2문장>",\n'
    '  "overall_vibe": "<전체 인상과 매력 3문장>",\n'
)


async def request_gemma_descriptions(
    user_image_url: str,
    matches: List[Dict[str, Any]],
    llama_url: str,
    model_name: str,
    mode: str = "full",
    first_rank: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Ask Gemma to describe the user's face and/or explain already chosen matches. None on failure.
    mode "top": user features + detailed #1 (fast); "others": brief write-ups only; "full": everything at once.
    Split calls keep each request well under the 60s the public reverse proxy allows.
    """
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": "[사진 0] 사용자:"},
        {"type": "image_url", "image_url": {"url": user_image_url}},
    ]
    for i, m in enumerate(matches, start=1):
        photo = _photo_data_url(m["entry"])
        if photo:
            content.append({"type": "text", "text": f"\n[사진 {i}] {m['entry']['name']} ({m['entry'].get('category', '연예인')}):"})
            content.append({"type": "image_url", "image_url": {"url": photo}})

    names = ", ".join(f"{first_rank + i}위 {m['entry']['name']}" for i, m in enumerate(matches))
    with_features = mode in ("top", "full")
    schema = {"others": BRIEF_MATCH_SCHEMA, "top": COMPACT_MATCH_SCHEMA}.get(mode, DETAILED_MATCH_SCHEMA)
    features_schema = COMPACT_FEATURES_SCHEMA if mode == "top" else USER_FEATURES_SCHEMA
    prompt = (
        "\n\n당신은 안면 형태학 전문가입니다. 얼굴 인식 AI가 [사진 0] 사용자와 닮은 연예인으로 "
        f"{names}을(를) 이미 선정했습니다.\n"
        "연예인 선정은 끝났으므로 이름을 바꾸거나 다른 연예인을 추천하지 마세요.\n"
        "사진을 직접 비교해서 실제로 보이는 공통점과 차이점을 구체적으로 쓰세요. 헤어스타일·옷·안경·표정 같은 외적 요소가 아니라 "
        "눈매(쌍꺼풀, 눈꼬리, 눈 크기), 코(콧대, 콧볼, 코끝), 입매(입술 두께, 입꼬리), 얼굴형(턱선, 광대, 얼굴 길이), "
        "이목구비 배치를 중심으로 설명합니다. '부드러운 인상' 같은 막연한 표현만 반복하지 말고 형태를 묘사하세요.\n"
        "detailed_scores는 각 부위가 얼마나 닮았는지 0~100 점수입니다. 부위 간 차이가 드러나도록 솔직하게 매기세요.\n\n"
        + "설명에서는 '사진 0', '사진 1' 같은 번호 대신 '사용자'와 연예인 이름으로 지칭하세요.\n"
        + ("글자 수 제한을 반드시 지키고, JSON은 들여쓰기·줄바꿈 없이 한 줄로 출력하세요.\n" if mode == "top" else "")
        + "반드시 다음 JSON 형식으로만 출력하세요:\n"
        "{\n"
        f"{features_schema if with_features else ''}"
        '  "matches": [\n'
        f"{schema}\n"
        "  ]\n"
        "}\n"
        f"matches 배열은 정확히 {len(matches)}개이며 {names} 순서대로 작성하세요."
    )
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.3,
        "max_tokens": {"top": 1000, "others": 1800}.get(mode, 3500),
        "stream": False,
        "cache_prompt": False,
    }
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=150.0) as client:
            resp = await client.post(f"{llama_url}/v1/chat/completions", json=payload,
                                     headers={"Content-Type": "application/json"})
        logger.info("gemma describe mode=%s matches=%d took %.1fs status=%s",
                    mode, len(matches), time.monotonic() - started, resp.status_code)
        if resp.status_code != 200:
            return None
        return extract_json_block(resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning(f"Gemma description call failed after {time.monotonic() - started:.1f}s: {e}")
        return None


def _format_match(rank: int, entry: Dict[str, Any], percent: int, cosine: Optional[float],
                  g: Dict[str, Any], pending: bool = False) -> Dict[str, Any]:
    """One result row. `g` is Gemma's description (may be empty); `pending` leaves texts blank for later."""
    if pending:
        summary = reason = ""
        points: List[str] = []
    else:
        summary = g.get("summary") or entry.get("face_type") or "얼굴 인식 AI가 측정한 눈·코·입 배치와 얼굴 윤곽이 가까운 후보입니다."
        reason = g.get("reason") or entry.get("vibe") or "눈매와 얼굴형 등 이목구비의 전체적인 배치가 유사합니다."
        points = g.get("matching_points") or [p for p in [entry.get("face_type"), "이목구비 배치", "얼굴 윤곽"] if p]
    return {
        "rank": rank,
        "qid": entry["qid"],
        "name": entry["name"],
        "category": entry.get("category", "연예인"),
        "similarity_percent": percent,
        "cosine": round(cosine, 4) if cosine is not None else None,
        "detailed_scores": adjust_part_scores(g.get("detailed_scores"), percent, PART_KEYS),
        "summary": summary,
        "reason": reason,
        "matching_points": [str(p) for p in points][:4],
        "part_notes": {k: str(v) for k, v in (g.get("part_notes") or {}).items()
                       if k in PART_KEYS and isinstance(v, (str, int, float))},
        "differences": [str(d) for d in (g.get("differences") or []) if d][:3],
        "photo_url": entry["photo_url"],
        "photo_source": entry.get("photo_source"),
    }


def _format_features(gemma: Dict[str, Any], pending: bool = False) -> Dict[str, Any]:
    missing = "" if pending else "분석 정보 없음"
    return {
        "gender": gemma.get("gender") or "",
        "age_group": gemma.get("age_group") or "",
        "keywords": [str(k) for k in (gemma.get("impression_keywords") or []) if k][:4],
        "face_type": gemma.get("animal_vibe") or missing,
        "face_shape": gemma.get("face_shape") or missing,
        "eyes": gemma.get("eyes") or missing,
        "nose": gemma.get("nose") or "",
        "mouth": gemma.get("mouth") or "",
        "nose_mouth": gemma.get("nose_mouth") or " ".join(x for x in (gemma.get("nose"), gemma.get("mouth")) if x) or missing,
        "overall_vibe": gemma.get("overall_vibe") or ("" if pending else "AI 설명을 생성하지 못해 얼굴 인식 결과만 표시합니다."),
    }


def _gemma_rows(gemma: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = gemma.get("matches") if isinstance(gemma.get("matches"), list) else []
    return [r if isinstance(r, dict) else {} for r in rows]


async def execute_celebrity_lookalike(
    image_b64: str,
    gender_filter: str = "auto",
    llama_url: str = "http://127.0.0.1:8081",
    model_name: str = "gemma-4-e4b-it-q4km",
    extra_images_b64: Optional[List[str]] = None,
    describe: bool = True,
) -> Dict[str, Any]:
    """
    Rank celebrities for the user's photo(s). With describe=False the ArcFace ranking returns in ~2s
    and the texts are left blank (`descriptions_pending`); the client then calls describe_celebrity_matches,
    because the detailed Gemma write-up for 5 matches takes ~50s.
    """
    raw_bytes = data_url_to_bytes(image_b64)

    entries, matrix = load_celebrity_db()
    if not entries:
        raise RuntimeError("연예인 데이터베이스가 비어 있습니다. populate_celebrities.py로 DB를 생성하세요.")

    main = extract_face_features(raw_bytes, with_visual=True)
    if main is None:
        raise FaceNotFoundError("사진에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 정면 사진을 올려 주세요.")
    main_feat, user_visual = main

    # Averaging several photos of the same person cancels out one photo's angle/lighting/expression
    # (look-alike eval: median rank of the known look-alike 36 -> 18 with 3 photos)
    feats, rejected = [main_feat], 0
    for extra in extra_images_b64 or []:
        # In group photos use the face that best matches the main photo, not the largest one
        try:
            match = best_matching_embedding(data_url_to_bytes(extra), main_feat)
        except Exception:
            match = None
        if match is not None and match[1] >= SAME_PERSON_MIN_COS:
            feats.append(match[0])
        else:
            rejected += 1
    user_feat = np.mean(feats, axis=0)
    user_feat /= np.linalg.norm(user_feat)

    # 1. Rank every celebrity by ArcFace cosine similarity
    sims = matrix @ user_feat.astype(np.float32)
    gender = gender_filter if gender_filter in ("male", "female") else infer_gender(sims, entries)
    pool = [i for i, e in enumerate(entries) if e.get("gender") == gender] or list(range(len(entries)))
    ranked, scores = visual_rerank(sims, pool, user_visual, load_visual_matrix(entries))
    top = ranked[:TOP_K]
    matches = [{"entry": entries[i], "cosine": float(sims[i]),
                "percent": similarity_to_percent(scores[i], "celebrity")} for i in top]
    logger.info(
        "celebrity search: gender=%s(%s) photos=%d rejected=%d visual=%s top=%s",
        gender, gender_filter, len(feats), rejected, user_visual is not None,
        ", ".join(f"{m['entry']['name']} {m['percent']}% (cos {m['cosine']:.3f})" for m in matches))

    # 2. Gemma writes the explanations for the fixed matches (now, or later via describe_celebrity_matches)
    gemma: Dict[str, Any] = {}
    if describe:
        user_url = image_bytes_to_data_url(raw_bytes, max_dim=768)
        gemma = await request_gemma_descriptions(user_url, matches[:GEMMA_DESCRIBED], llama_url, model_name) or {}
    rows = _gemma_rows(gemma)
    formatted = [
        _format_match(rank, m["entry"], m["percent"], m["cosine"],
                      rows[rank - 1] if rank - 1 < len(rows) else {}, pending=not describe)
        for rank, m in enumerate(matches, start=1)
    ]

    return {
        "success": True,
        "method": "arcface",
        "gender_used": gender,
        "photos_used": len(feats),
        "photos_rejected": rejected,
        "descriptions_pending": not describe,
        "face_features": _format_features(gemma, pending=not describe),
        "match_strength": celebrity_match_strength(matches[0]["cosine"]) if matches else None,
        "top_celebrity": formatted[0] if formatted else None,
        "candidates": formatted[1:],
        "all_celebrities": formatted,
    }


async def describe_celebrity_matches(
    image_b64: str,
    qids: List[str],
    percents: List[int],
    llama_url: str,
    model_name: str,
    part: str = "all",
) -> Dict[str, Any]:
    """
    Second phase: Gemma descriptions for matches already chosen by execute_celebrity_lookalike.
    part "top" -> my face features + detailed #1; "others" -> brief #2..#5; "all" -> everything (slow).
    """
    entries, _ = load_celebrity_db()
    by_qid = {e["qid"]: e for e in entries}
    ranked = [(i, by_qid[q]) for i, q in enumerate(qids[:GEMMA_DESCRIBED]) if q in by_qid]
    if part == "top":
        ranked, mode = ranked[:1], "top"
    elif part == "others":
        ranked, mode = ranked[1:], "others"
    else:
        mode = "full"
    if not ranked:
        raise ValueError("설명할 연예인 정보가 없습니다.")
    user_url = image_bytes_to_data_url(data_url_to_bytes(image_b64), max_dim=768)
    gemma = await request_gemma_descriptions(user_url, [{"entry": e} for _, e in ranked], llama_url, model_name,
                                             mode=mode, first_rank=ranked[0][0] + 1)
    if not gemma:
        return {"success": False}
    rows = _gemma_rows(gemma)
    return {
        "success": True,
        "face_features": _format_features(gemma) if mode != "others" else None,
        "matches": [
            _format_match(i + 1, e, int(percents[i]) if i < len(percents) else 70, None, rows[n] if n < len(rows) else {})
            for n, (i, e) in enumerate(ranked)
        ],
    }
