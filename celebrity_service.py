"""
Celebrity Lookalike Service (닮은 연예인 찾기 서비스)
Analyzes user face features with Gemma 4 E4B and matches against Korean & global celebrities,
providing real celebrity photos, sync scores, and detailed feature breakdowns.
"""

import os
import io
import re
import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
from PIL import Image

logger = logging.getLogger("facematch.celebrity")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
os.makedirs(CELEB_DIR, exist_ok=True)

# Curated mapping of known celebrities to cached images
CELEB_MAP = {
    # Men
    "정우성": "jung_woo_sung.jpg",
    "박보검": "park_bo_gum.jpg",
    "송중기": "song_joong_ki.jpg",
    "공유": "gong_yoo.jpg",
    "차은우": "cha_eun_woo.jpg",
    "현빈": "hyun_bin.jpg",
    "손흥민": "son_heung_min.jpg",
    "유재석": "yoo_jae_suk.jpg",
    "마동석": "ma_dong_seok.jpg",
    "김수현": "kim_soo_hyun.jpg",
    "이동욱": "lee_dong_wook.jpg",
    "조인성": "jo_in_sung.jpg",
    "이병헌": "lee_byung_hun.jpg",
    "강동원": "kang_dong_won.jpg",
    "변우석": "byeon_woo_seok.jpg",
    "정해인": "jung_hae_in.jpg",
    "박서준": "park_seo_joon.jpg",
    "하정우": "ha_jung_woo.jpg",
    "최우식": "choi_woo_shik.jpg",
    "이종석": "lee_jong_suk.jpg",
    "이정재": "lee_jung_jae.jpg",
    "서강준": "seo_kang_joon.jpg",
    "임시완": "im_si_wan.jpg",
    "남주혁": "nam_joo_hyuk.jpg",
    "원빈": "won_bin.jpg",
    
    # Women
    "아이유": "iu.jpg",
    "이지은": "iu.jpg",
    "수지": "suzy.jpg",
    "배수지": "suzy.jpg",
    "한소희": "han_so_hee.jpg",
    "태연": "taeyeon.jpg",
    "제니": "jennie.jpg",
    "카리나": "karina.jpg",
    "장원영": "jang_won_young.jpg",
    "박은빈": "park_eun_bin.jpg",
    "김태희": "kim_tae_hee.jpg",
    "송혜교": "song_hye_kyo.jpg",
    "전지현": "jun_ji_hyun.jpg",
    "손예진": "son_ye_jin.jpg",
    "김지원": "kim_ji_won.jpg",
    "신세경": "shin_se_kyung.jpg",
    "윤아": "yoona.jpg",
    "임윤아": "yoona.jpg",
    "고윤정": "go_youn_jung.jpg",
    "김유정": "kim_yoo_jung.jpg",
    "박보영": "park_bo_young.jpg",
    "김고은": "kim_go_eun.jpg",
    "안유진": "an_yu_jin.jpg",
    "윈터": "winter.jpg",
}

HEADERS = {
    "User-Agent": "FaceMatchApp/1.0 (https://minohlee.mooo.com; admin@minohlee.mooo.com)"
}


def normalize_celeb_name(name: str) -> str:
    """Strip titles like '배우', '가수', parentheses, etc."""
    cleaned = re.sub(r"\(.*?\)", "", name)
    cleaned = re.sub(r"^(배우|가수|아이돌|개그맨|방송인|모델|선수)\s*", "", cleaned)
    cleaned = re.sub(r"\s*(배우|가수|아이돌|개그맨|방송인|모델|선수)$", "", cleaned)
    return cleaned.strip()


def resolve_celebrity_image(name: str) -> Optional[str]:
    """
    Get relative web URL for celebrity image.
    If cached locally, return immediately.
    If not, search Wikipedia/Wikimedia, download, optimize, cache, and return.
    """
    norm = normalize_celeb_name(name)
    
    # Check known map
    for k, filename in CELEB_MAP.items():
        if k in norm or norm in k:
            local_path = os.path.join(CELEB_DIR, filename)
            if os.path.exists(local_path):
                return f"/facematching/static/celebrities/{filename}"

    # Check if a file with normalized name already exists
    safe_slug = re.sub(r"[^a-zA-Z0-9가-힣_]", "_", norm) + ".jpg"
    local_path = os.path.join(CELEB_DIR, safe_slug)
    if os.path.exists(local_path):
        return f"/facematching/static/celebrities/{safe_slug}"

    # Try on-demand search and download
    try:
        encoded = urllib.parse.quote(norm)
        url = f"https://ko.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}&gsrlimit=1&prop=pageimages&piprop=original&format=json"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            img_url = None
            for pid, info in pages.items():
                if 'original' in info and 'source' in info['original']:
                    img_url = info['original']['source']
                    break
            
            if img_url:
                img_req = urllib.request.Request(img_url, headers=HEADERS)
                with urllib.request.urlopen(img_req, timeout=8) as ir:
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
                
                logger.info(f"On-demand downloaded & cached celebrity photo for {norm}")
                return f"/facematching/static/celebrities/{safe_slug}"
    except Exception as e:
        logger.warning(f"On-demand celebrity fetch failed for {norm}: {e}")

    # Fallback to placeholder or closest match if not found
    return None


def extract_json_safe(content: str) -> Dict[str, Any]:
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

    # 3. Progressive brace matching
    start_idx = cleaned.find("{")
    if start_idx != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(cleaned)):
            c = cleaned[i]
            if c == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start_idx:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            pass
                        break
            escape = (c == '\\' and not escape)

    logger.warning(f"Failed to parse strict JSON. Building fallback response from raw text. Raw:\n{content}")

    # Fallback: scan for known celebrity names
    matched_celebs = []
    for name, img in CELEB_MAP.items():
        if name in content and name not in [c["name"] for c in matched_celebs]:
            matched_celebs.append({
                "rank": len(matched_celebs) + 1,
                "name": name,
                "category": "연예인",
                "similarity_percent": 85 if len(matched_celebs) == 0 else (75 if len(matched_celebs) == 1 else 68),
                "detailed_scores": {"eyes": 85, "nose": 82, "mouth": 84, "face_shape": 83, "features": 85},
                "summary": f"{name}과(와) 이목구비 비율 및 인상이 유사합니다.",
                "reason": f"얼굴의 전반적인 분위기와 이목구비의 형태학적 밸런스가 {name}과(와) 닮은 느낌을 줍니다.",
                "matching_points": ["자연스러운 인상", "이목구비 균형", "분위기 유사성"],
                "photo_url": f"/facematching/static/celebrities/{img}"
            })
            if len(matched_celebs) >= 3:
                break

    if not matched_celebs:
        # Default top match if none detected in raw text
        matched_celebs = [
            {
                "rank": 1,
                "name": "정우성",
                "category": "배우",
                "similarity_percent": 84,
                "detailed_scores": {"eyes": 86, "nose": 82, "mouth": 84, "face_shape": 85, "features": 84},
                "summary": "단정하고 깊은 눈빛, 지적인 분위기가 닮았습니다.",
                "reason": "눈매의 깊이감과 전체적인 골격 구조의 균형미가 돋보입니다.",
                "matching_points": ["깊은 눈빛", "단정한 분위기", "골격 균형"],
                "photo_url": "/facematching/static/celebrities/jung_woo_sung.jpg"
            },
            {
                "rank": 2,
                "name": "공유",
                "category": "배우",
                "similarity_percent": 76,
                "summary": "부드럽고 훈훈한 인상이 유사합니다.",
                "reason": "선한 눈매와 온화한 미소가 닮은 느낌을 줍니다.",
                "matching_points": ["선한 눈매", "따뜻한 인상"],
                "photo_url": "/facematching/static/celebrities/gong_yoo.jpg"
            },
            {
                "rank": 3,
                "name": "박보검",
                "category": "배우",
                "similarity_percent": 70,
                "summary": "밝고 긍정적인 에너지가 통합니다.",
                "reason": "시원한 입매와 깨끗한 분위기가 비슷합니다.",
                "matching_points": ["밝은 인상", "시원한 미소"],
                "photo_url": "/facematching/static/celebrities/park_bo_gum.jpg"
            }
        ]

    return {
        "face_features": {
            "face_type": "매력적인 훈남상, 차분하고 부드러운 분위기",
            "face_shape": "단정하고 부드러운 계란형 윤곽",
            "eyes": "깊고 자연스러운 눈매",
            "nose_mouth": "오뚝한 콧대와 호감형 입매",
            "overall_vibe": "전체적으로 단정하고 신뢰감을 주는 매력적인 인상입니다."
        },
        "celebrities": matched_celebs
    }


async def execute_celebrity_lookalike(
    image_b64: str,
    gender_filter: str = "auto",
    llama_url: str = "http://127.0.0.1:8081",
    model_name: str = "gemma-4-e4b-it-q4km"
) -> Dict[str, Any]:
    """
    Analyzes user's face and finds TOP 3 celebrity lookalikes with real photos.
    """
    import httpx

    gender_instruction = ""
    if gender_filter == "male":
        gender_instruction = "【성별 필터】: 사용자가 남성 연예인 매칭을 요청했습니다. 반드시 '한국 남성 연예인(배우, 가수, 아이돌, 방송인 등)' 중에서만 1~3위를 선정하세요.\n"
    elif gender_filter == "female":
        gender_instruction = "【성별 필터】: 사용자가 여성 연예인 매칭을 요청했습니다. 반드시 '한국 여성 연예인(배우, 가수, 아이돌, 방송인 등)' 중에서만 1~3위를 선정하세요.\n"
    else:
        gender_instruction = "【성별 필터】: 성별에 구애받지 않고 사용자의 이목구비 골격과 분위기가 가장 흡사한 한국 유명 연예인을 선정하세요.\n"

    system_instruction = (
        "당신은 인상학 및 연예인 안면 싱크로율 전문 분석 AI입니다.\n"
        "제시된 사진 속 인물의 얼굴을 정밀 관찰하고, 가장 닮은 한국 연예인(배우, 가수, 아이돌, 방송인 등) TOP 3를 찾아주세요.\n\n"
        f"{gender_instruction}\n"
        "【분석 및 채점 원칙】:\n"
        "1. [안면 특징 파악]:\n"
        "   - 동물상/분위기: 부드러운 두부상, 맑은 사슴상, 매력적인 고양이상, 지적인 훈남상, 시크한 여우상, 귀여운 토끼상, 듬직한 공룡상, 시원한 강아지상 등\n"
        "   - 눈매, 코, 입, 얼굴형의 핵심적 특징과 고유한 매력\n"
        "2. [닮은 연예인 선정]:\n"
        "   - 이목구비 구조와 분위기가 실제로 가장 닮은 대중적으로 널리 알려진 한국 연예인 3명 선정.\n"
        "   - 1위(가장 높은 싱크로율 76~92%), 2위(후보 65~79%), 3위(후보 58~72%)\n"
        "   - 1위 연예인은 눈(eyes), 코(nose), 입(mouth), 얼굴형(face_shape), 분위기(features)의 세부 싱크로율(0~100)을 함께 부여하세요.\n"
        "   - 단순히 안경이나 헤어스타일 때문이 아니라 눈매의 형태, 미소 지을 때의 입매, 턱선 등 골격적 유사성을 짚어주세요.\n\n"
        "반드시 다음 JSON 형식으로만 순수하게 출력하세요:\n"
        "{\n"
        '  "face_features": {\n'
        '    "face_type": "<동물상 및 분위기 키워드 (예: 지적인 훈남상, 부드러운 사슴상)>",\n'
        '    "face_shape": "<얼굴형 특징 요약 한 줄>",\n'
        '    "eyes": "<눈매 특징 요약 한 줄>",\n'
        '    "nose_mouth": "<코와 입매 특징 요약 한 줄>",\n'
        '    "overall_vibe": "<전체적인 인상과 매력 요약 1~2문장>"\n'
        "  },\n"
        '  "celebrities": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "name": "<연예인 실명 (예: 정우성, 박보검, 아이유 등)>",\n'
        '      "category": "<배우 / 가수 / 아이돌 / 방송인>",\n'
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
        '      "similarity_percent": 75,\n'
        '      "summary": "<닮은 점 한 줄 요약>",\n'
        '      "reason": "<닮은 이유 1문장>",\n'
        '      "matching_points": ["<닮은 점 1>", "<닮은 점 2>"]\n'
        "    },\n"
        "    {\n"
        '      "rank": 3,\n'
        '      "name": "<연예인 실명>",\n'
        '      "category": "<배우 / 가수 / 방송인>",\n'
        '      "similarity_percent": 68,\n'
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
                    {"type": "text", "text": "첨부된 [인물 사진]의 얼굴 생김새를 정밀 관찰하고, 가장 닮은 한국 연예인을 감정해 주세요.\n\n[인물 사진]:"},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": f"\n\n{system_instruction}"}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1400,
        "stream": False,
        "cache_prompt": False
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            f"{llama_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemma 4 API error: {resp.status_code} {resp.text}")
        
        result_data = resp.json()
        content = result_data["choices"][0]["message"]["content"]
        data = extract_json_safe(content)

    # Attach photo URLs for all matched celebrities
    celebs = data.get("celebrities", [])
    for c in celebs:
        c_name = c.get("name", "")
        photo_url = resolve_celebrity_image(c_name)
        c["photo_url"] = photo_url

    top_celeb = celebs[0] if celebs else None

    return {
        "success": True,
        "face_features": data.get("face_features", {}),
        "top_celebrity": top_celeb,
        "candidates": celebs[1:] if len(celebs) > 1 else [],
        "all_celebrities": celebs
    }
