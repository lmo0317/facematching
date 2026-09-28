"""
Build celebrity_db.json: Korean celebrities with a multi-photo ArcFace embedding each.

1. Candidates = seed CELEBRITY_CATALOG below (hand-written descriptions) + living South Korean
   entertainers from Wikidata that have a Wikimedia Commons category, ranked by Wikipedia sitelinks.
2. Up to PHOTOS_PER_PERSON photos per person are pulled from Commons (deepcat search) and cached
   under data/celeb_cache/<QID>/.
3. The person's face is identified by consensus: the face that recurs across their photos wins,
   so group shots, logos and other people are discarded. Matching faces are averaged into one
   normalized embedding.
4. A face-centered square display photo is written to static/celebrities/.

Usage: python populate_celebrities.py [--limit 600] [--workers 4]
Requires models/face_detection_yunet.onnx and models/arcface_w600k_r50.onnx.
"""

import os
import re
import json
import time
import argparse
import threading
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

from face_utils import (
    ARCFACE_PATH, YUNET_PATH, HTTP_HEADERS,
    compute_square_face_box, decode_image_bgr, detect_faces_yunet, embed_face,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
CACHE_DIR = os.path.join(BASE_DIR, "data", "celeb_cache")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")

PHOTOS_PER_PERSON = 10
THUMB_WIDTH = 960
MIN_FACE_PX = 60
SAME_PERSON_COS = 0.40   # ArcFace: different people rarely exceed ~0.3
DISPLAY_SIZE = 360

GENDER_QIDS = {"Q6581097": "male", "Q6581072": "female"}
OCCUPATION_CATEGORY = [  # first match wins
    ({"Q177220", "Q488205"}, "가수"),
    ({"Q33999", "Q10800557", "Q10798782"}, "배우"),
    ({"Q947873", "Q245068"}, "방송인"),
    ({"Q4610556"}, "모델"),
]

os.makedirs(CELEB_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Hand-picked seeds: always included, and their descriptions are reused in result text.
CELEBRITY_CATALOG = [
    # --- MEN ---
    # 조각 / 정석 미남형
    {"name": "정우성", "query": "정우성", "file": "jung_woo_sung.jpg", "gender": "male", "category": "배우", "face_type": "짙은 눈썹과 깊은 눈빛의 클래식 조각상", "vibe": "중후하고 깊이 있는 눈매, 완벽한 T존과 신뢰감 넘치는 귀족적 아우라"},
    {"name": "강동원", "query": "강동원", "file": "kang_dong_won.jpg", "gender": "male", "category": "배우", "face_type": "날렵하고 신비로운 늑대·사슴상", "vibe": "동양적이면서도 만화 같은 비현실적인 비율과 오뚝한 콧대"},
    {"name": "원빈", "query": "원빈", "file": "won_bin.jpg", "gender": "male", "category": "배우", "face_type": "전설적인 황금비율 조각미남", "vibe": "우수에 찬 눈빛과 흠잡을 데 없는 이목구비 밸런스"},
    {"name": "조인성", "query": "조인성 (배우)", "file": "jo_in_sung.jpg", "gender": "male", "category": "배우", "face_type": "높은 콧대와 날렵한 턱선의 미남상", "vibe": "시원시원한 마스크와 세련되고 카리스마 넘치는 분위기"},
    {"name": "차은우", "query": "차은우", "file": "cha_eun_woo.jpg", "gender": "male", "category": "아이돌", "face_type": "비현실적인 만찢남 비주얼", "vibe": "맑고 큰 눈망울, 또렷한 티존과 완벽한 안면 대칭"},
    {"name": "현빈", "query": "현빈", "file": "hyun_bin.jpg", "gender": "male", "category": "배우", "face_type": "선 굵은 턱선과 보조개의 정석 훈남", "vibe": "듬직하면서도 부드러운 눈매, 신뢰감을 주는 귀공자형 인상"},
    {"name": "이동욱", "query": "이동욱 (배우)", "file": "lee_dong_wook.jpg", "gender": "male", "category": "배우", "face_type": "깊은 아이홀과 시크한 뱀파이어상", "vibe": "이국적인 눈매와 붉고 매력적인 입술선, 차분한 분위기"},
    {"name": "송승헌", "query": "송승헌", "file": "song_seung_heon.jpg", "gender": "male", "category": "배우", "face_type": "숯검댕이 눈썹과 또렷한 이목구비", "vibe": "클래식하고 남자다운 마스크와 강인한 눈매"},
    
    # 부드러운 훈남 / 사슴·두부상
    {"name": "박보검", "query": "박보검", "file": "park_bo_gum.jpg", "gender": "male", "category": "배우", "face_type": "맑은 사슴상, 시원한 미소의 훈남", "vibe": "선하고 반짝이는 눈망울, 환한 미소와 다정한 분위기"},
    {"name": "송중기", "query": "송중기", "file": "song_joong_ki.jpg", "gender": "male", "category": "배우", "face_type": "뽀얗고 부드러운 밀크 두부상", "vibe": "청량하고 소년미 넘치는 눈매와 단정한 입매"},
    {"name": "정해인", "query": "정해인", "file": "jung_hae_in.jpg", "gender": "male", "category": "배우", "face_type": "깨끗하고 단정한 선비·밀크남", "vibe": "깔끔한 피부톤과 온화한 눈빛, 신뢰감을 주는 미소"},
    {"name": "임시완", "query": "임시완", "file": "im_si_wan.jpg", "gender": "male", "category": "배우", "face_type": "단정하고 총명한 맑은 인상", "vibe": "섬세하고 반듯한 이목구비, 맑고 깨끗한 눈매"},
    {"name": "변우석", "query": "변우석", "file": "byeon_woo_seok.jpg", "gender": "male", "category": "배우", "face_type": "청량한 소년미와 날렵한 턱선", "vibe": "웃을 때 시원하게 열리는 입매와 맑은 눈망울"},
    {"name": "서강준", "query": "서강준", "file": "seo_kang_joon.jpg", "gender": "male", "category": "배우", "face_type": "신비로운 갈색 눈동자 미남", "vibe": "조각 같은 콧날과 서정적인 멜로 눈빛"},
    {"name": "남주혁", "query": "남주혁", "file": "nam_joo_hyuk.jpg", "gender": "male", "category": "배우", "face_type": "소년미 넘치는 훈훈한 멍뭉이상", "vibe": "세련된 하관과 쌍꺼풀 없이 담백하고 시원한 눈매"},

    # 매력적인 무쌍 / 개성 / 샤프 공룡·여우상
    {"name": "공유", "query": "공유 (배우)", "file": "gong_yoo.jpg", "gender": "male", "category": "배우", "face_type": "듬직하고 따뜻한 공룡상", "vibe": "깊고 온화한 눈매, 듬직한 골격과 매력적인 미소"},
    {"name": "박서준", "query": "박서준", "file": "park_seo_joon.jpg", "gender": "male", "category": "배우", "face_type": "훈훈하고 스타일리시한 매력 무쌍", "vibe": "시원한 눈매와 웃을 때 반달이 되는 눈웃음, 탄탄한 하관"},
    {"name": "김수현", "query": "김수현 (1988년)", "file": "kim_soo_hyun.jpg", "gender": "male", "category": "배우", "face_type": "작은 얼굴과 올라간 입꼬리의 매력남", "vibe": "짙은 눈썹과 또렷한 눈매, 위로 솟은 매력적인 입꼬리"},
    {"name": "이종석", "query": "이종석 (배우)", "file": "lee_jong_suk.jpg", "gender": "male", "category": "배우", "face_type": "도톰한 입술과 뽀얀 피부의 소년미", "vibe": "귀여운 눈밑 점, 도톰하고 입체적인 입술 라인"},
    {"name": "최우식", "query": "최우식", "file": "choi_woo_shik.jpg", "gender": "male", "category": "배우", "face_type": "친근하고 순둥순둥한 강아지상", "vibe": "순하고 편안한 눈매, 보호본능을 자극하는 귀여운 인상"},
    {"name": "류준열", "query": "류준열", "file": "ryu_jun_yeol.jpg", "gender": "male", "category": "배우", "face_type": "독보적인 무쌍 눈매와 시크한 매력", "vibe": "동양적인 매력이 돋보이는 날렵한 눈매와 트렌디한 마스크"},
    {"name": "손석구", "query": "손석구", "file": "son_suk_ku.jpg", "gender": "male", "category": "배우", "face_type": "치명적인 나른한 눈빛의 여우·늑대상", "vibe": "무쌍의 깊은 눈매, 날렵한 턱선과 섹시하고 거친 아우라"},
    {"name": "김우빈", "query": "김우빈", "file": "kim_woo_bin.jpg", "gender": "male", "category": "배우", "face_type": "카리스마 넘치는 대표 공룡상", "vibe": "짙은 눈썹, 강렬한 T존과 시원시원한 입매"},
    {"name": "이도현", "query": "이도현 (배우)", "en_query": "Lee Do-hyun", "file": "lee_do_hyun.jpg", "gender": "male", "category": "배우", "face_type": "매력적인 입꼬리와 맑고 깊은 눈", "vibe": "청량한 소년미와 성숙한 카리스마가 공존하는 마스크"},

    # 듬직 / 선 굵은 카리스마 / 상남자형
    {"name": "이정재", "query": "이정재", "en_query": "Lee Jung-jae", "file": "lee_jung_jae.jpg", "gender": "male", "category": "배우", "face_type": "매력적인 광대와 기품 있는 미소", "vibe": "고급스럽고 중후한 매력, 웃을 때 번지는 눈웃음과 턱선"},
    {"name": "하정우", "query": "하정우", "file": "ha_jung_woo.jpg", "gender": "male", "category": "배우", "face_type": "선 굵고 묵직한 카리스마", "vibe": "남성미 넘치는 턱선과 짙은 눈빛, 거침없는 아우라"},
    {"name": "이병헌", "query": "이병헌", "file": "lee_byung_hun.jpg", "gender": "male", "category": "배우", "face_type": "강렬하고 묵직한 하관과 깊은 눈빛", "vibe": "단단한 턱선, 압도적인 눈빛과 신뢰감을 주는 골격"},
    {"name": "마동석", "query": "마동석", "file": "ma_dong_seok.jpg", "gender": "male", "category": "배우", "face_type": "다부지고 듬직한 베어(곰)상", "vibe": "묵직하고 단단한 하관과 푸근하면서도 압도적인 카리스마"},
    {"name": "황정민", "query": "황정민 (배우)", "en_query": "Hwang Jung-min", "file": "hwang_jung_min.jpg", "gender": "male", "category": "배우", "face_type": "사람 냄새 나는 따뜻하고 깊은 마스크", "vibe": "진솔한 눈매와 친근한 주름, 진정성 넘치는 인상"},
    {"name": "조진웅", "query": "조진웅", "file": "cho_jin_woong.jpg", "gender": "male", "category": "배우", "face_type": "듬직하고 웅장한 호랑이상", "vibe": "묵직한 안면 골격과 신뢰감을 주는 중후한 매력"},

    # 친근 / 호감 / 푸근 / 동글형
    {"name": "안재홍", "query": "안재홍 (배우)", "file": "ahn_jae_hong.jpg", "gender": "male", "category": "배우", "face_type": "순박하고 정감 가는 순둥이 호감형", "vibe": "동글동글한 콧망울과 온화한 눈매, 보는 이를 편안하게 하는 인상"},
    {"name": "조세호", "query": "조세호", "file": "cho_sae_ho.jpg", "gender": "male", "category": "방송인", "face_type": "동글동글 푸근한 복덩이 미소", "vibe": "통통하고 귀여운 볼살, 친근하고 유쾌한 에너지"},
    {"name": "싸이", "query": "싸이 (가수)", "file": "psy.jpg", "gender": "male", "category": "가수", "face_type": "에너지 넘치고 개성 있는 둥근 얼굴", "vibe": "자신감 넘치는 눈빛과 시원한 미소, 독보적인 캐릭터"},
    {"name": "유재석", "query": "유재석", "file": "yoo_jae_suk.jpg", "gender": "male", "category": "방송인", "face_type": "친근하고 밝은 국민 MC상", "vibe": "선하고 지적인 눈매, 시원한 입매와 신뢰감을 주는 표정"},
    {"name": "신동엽", "query": "신동엽 (방송인)", "en_query": "Shin Dong-yup", "file": "shin_dong_yup.jpg", "gender": "male", "category": "방송인", "face_type": "오밀조밀 모인 개성 만점 호감상", "vibe": "장난기 넘치는 눈웃음과 재치 있는 인상"},
    {"name": "기안84", "query": "기안84", "file": "kian84.jpg", "gender": "male", "category": "방송인", "face_type": "날것 그대로의 순수하고 호탕한 인상", "vibe": "꾸밈없는 담백한 눈매와 시원털털한 미소"},
    {"name": "손흥민", "query": "손흥민", "file": "son_heung_min.jpg", "gender": "male", "category": "스포츠", "face_type": "기분 좋아지는 반달 눈웃음의 승부사", "vibe": "동양적인 매력의 무쌍 눈매와 건강하고 활기찬 에너지"},
    {"name": "임영웅", "query": "임영웅", "file": "lim_young_woong.jpg", "gender": "male", "category": "가수", "face_type": "따뜻하고 훈훈한 국민 힐러상", "vibe": "선한 눈망울과 차분한 입매, 신뢰와 위로를 주는 인상"},
    {"name": "김종국", "query": "김종국 (가수)", "file": "kim_jong_kook.jpg", "gender": "male", "category": "가수", "face_type": "호랑이상 귀여운 눈웃음", "vibe": "작고 매력적인 눈웃음과 듬직한 피지컬의 반전 매력"},
    {"name": "비", "query": "비 (가수)", "file": "rain.jpg", "gender": "male", "category": "가수", "face_type": "원조 매력 무쌍 눈매와 섹시한 턱선", "vibe": "날렵한 무쌍 눈매, 시원한 콧날과 독보적인 무대 아우라"},
    {"name": "문세윤", "query": "문세윤", "file": "moon_se_yoon.jpg", "gender": "male", "category": "방송인", "face_type": "푸근하고 복스러운 둥근 호감형", "vibe": "환하고 귀여운 눈웃음과 보기만 해도 기분 좋은 푸근함"},
    {"name": "이수근", "query": "이수근 (희극인)", "file": "lee_soo_geun.jpg", "gender": "male", "category": "방송인", "face_type": "익살스럽고 친근한 재치만점상", "vibe": "눈꼬리가 처진 선한 눈매와 친근한 미소"},
    {"name": "데프콘", "query": "데프콘 (가수)", "file": "defconn.jpg", "gender": "male", "category": "방송인", "face_type": "개성 넘치고 푸근한 호감형 마스크", "vibe": "둥글둥글한 안면 골격과 듬직하고 호탕한 인상"},
    {"name": "박명수", "query": "박명수", "file": "park_myung_soo.jpg", "gender": "male", "category": "방송인", "face_type": "독보적 개성의 거성 마스크", "vibe": "쌍꺼풀 짙은 눈매와 유쾌한 개성이 돋보이는 독보적 인상"},
    {"name": "뷔", "query": "V (singer)", "lang": "en", "file": "v_bts.jpg", "gender": "male", "category": "아이돌", "face_type": "비현실적인 K-POP 대표 조각 비주얼", "vibe": "크고 깊은 눈망울, 높은 콧대와 화려한 이목구비"},
    {"name": "정국", "query": "Jungkook", "lang": "en", "file": "jungkook.jpg", "gender": "male", "category": "아이돌", "face_type": "맑고 또렷한 순수 토끼상", "vibe": "동그랗고 반짝이는 눈망울과 또렷한 입체감"},
    {"name": "지민", "query": "Jimin", "lang": "en", "file": "jimin.jpg", "gender": "male", "category": "아이돌", "face_type": "동양적이고 매혹적인 무쌍 눈매", "vibe": "도톰하고 매력적인 입술, 부드러운 턱선과 몽환적 아우라"},
    {"name": "성시경", "query": "성시경", "file": "sung_si_kyung.jpg", "gender": "male", "category": "가수", "face_type": "지적이고 부드러운 발라드 귀공자", "vibe": "선한 눈매와 단정한 입매, 감미롭고 지적인 분위기"},

    # --- WOMEN ---
    # 청순 / 꽃사슴 / 맑은 사슴상
    {"name": "아이유", "query": "아이유", "file": "iu.jpg", "gender": "female", "category": "가수", "face_type": "맑고 투명한 강아지·사슴상", "vibe": "동글동글하고 맑은 눈망울, 작고 예쁜 콧망울과 사랑스러운 인상"},
    {"name": "수지", "query": "수지 (가수)", "file": "suzy.jpg", "gender": "female", "category": "배우", "face_type": "국민 첫사랑 청순 토끼상", "vibe": "맑고 시원한 눈매, 탐스러운 입술과 환하고 사랑스러운 미소"},
    {"name": "윤아", "query": "윤아", "file": "yoona.jpg", "gender": "female", "category": "가수", "face_type": "단아하고 맑은 대표 꽃사슴상", "vibe": "가녀리고 긴 목선, 크고 맑은 눈망울과 청순한 미소"},
    {"name": "박은빈", "query": "박은빈", "file": "park_eun_bin.jpg", "gender": "female", "category": "배우", "face_type": "총명하고 단아한 토끼상", "vibe": "반짝이는 맑은 눈망울, 정갈하고 사랑스러운 미소"},
    {"name": "손예진", "query": "손예진", "file": "son_ye_jin.jpg", "gender": "female", "category": "배우", "face_type": "눈웃음의 대명사 청순 여신", "vibe": "웃을 때 반달이 되는 사랑스러운 눈매와 맑은 피부"},
    {"name": "한효주", "query": "한효주", "file": "han_hyo_joo.jpg", "gender": "female", "category": "배우", "face_type": "투명하고 자연스러운 힐링 미인", "vibe": "시원시원한 입매와 깨끗한 눈빛, 온화한 아우라"},
    {"name": "신세경", "query": "신세경", "file": "shin_se_kyung.jpg", "gender": "female", "category": "배우", "face_type": "그윽한 눈매와 오뚝한 콧날의 분위기 여신", "vibe": "우아한 콧대와 깊은 눈빛, 차분하고 고급스러운 인상"},

    # 정석 미인 / 도회적 / 우아함
    {"name": "김태희", "query": "김태희", "file": "kim_tae_hee.jpg", "gender": "female", "category": "배우", "face_type": "황금비율 정석 미인의 표준", "vibe": "완벽한 안면 비례, 또렷하고 입체적인 눈코입"},
    {"name": "송혜교", "query": "송혜교", "file": "song_hye_kyo.jpg", "gender": "female", "category": "배우", "face_type": "우아하고 고혹적인 도화살 눈매", "vibe": "도톰하고 매력적인 입술, 깊고 그윽한 눈매"},
    {"name": "전지현", "query": "전지현", "file": "jun_ji_hyun.jpg", "gender": "female", "category": "배우", "face_type": "시원시원하고 독보적인 아우라", "vibe": "자연스러운 코 옆 점, 시원한 이목구비와 세련된 카리스마"},
    {"name": "김지원", "query": "김지원 (배우)", "file": "kim_ji_won.jpg", "gender": "female", "category": "배우", "face_type": "또렷하고 도회적인 세련 미인", "vibe": "크고 깊은 눈망울, 오뚝한 콧날과 단아하면서도 도도한 매력"},
    {"name": "고윤정", "query": "고윤정", "file": "go_youn_jung.jpg", "gender": "female", "category": "배우", "face_type": "흠잡을 데 없는 완벽한 T존 미인", "vibe": "맑고 또렷한 눈망울, 오뚝하고 세련된 콧대와 턱선"},

    # 화려 / 고양이 / 여우상 / 트렌디
    {"name": "한소희", "query": "한소희", "file": "han_so_hee.jpg", "gender": "female", "category": "배우", "face_type": "매혹적이고 시크한 고양이상", "vibe": "오뚝한 콧날, 도도하면서도 매혹적인 눈빛과 독보적 분위기"},
    {"name": "제니", "query": "제니 (가수)", "file": "jennie.jpg", "gender": "female", "category": "가수", "face_type": "트렌디한 힙 베이비 고양이상", "vibe": "도톰한 입술, 몽환적이고 매력적인 눈매와 베이비페이스"},
    {"name": "카리나", "query": "카리나 (가수)", "file": "karina.jpg", "gender": "female", "category": "아이돌", "face_type": "비현실적인 AI 달걀형 얼굴", "vibe": "작은 얼굴에 꽉 찬 큰 눈과 높은 코, 완벽한 브이라인"},
    {"name": "장원영", "query": "장원영", "file": "jang_won_young.jpg", "gender": "female", "category": "아이돌", "face_type": "화려하고 러블리한 체리 토끼상", "vibe": "도톰한 하트 입술, 큰 눈망울과 화려한 과즙미"},
    {"name": "태연", "query": "태연", "file": "taeyeon.jpg", "gender": "female", "category": "가수", "face_type": "오밀조밀 요정 같은 비주얼", "vibe": "하얗고 투명한 피부, 맑은 눈매와 섬세한 입매"},
    {"name": "윈터", "query": "윈터 (가수)", "file": "winter.jpg", "gender": "female", "category": "아이돌", "face_type": "날렵하고 또렷한 강아지·사슴상", "vibe": "하얀 피부와 앙증맞은 코, 또렷하고 깨끗한 눈매"},
    {"name": "안유진", "query": "안유진", "file": "an_yu_jin.jpg", "gender": "female", "category": "아이돌", "face_type": "시원시원하고 건강한 사슴상", "vibe": "크고 시원한 눈망울, 밝고 긍정적인 비타민 에너지"},

    # 매력적인 무쌍 / 자연미 / 러블리 / 개성
    {"name": "김고은", "query": "김고은", "file": "kim_go_eun.jpg", "gender": "female", "category": "배우", "face_type": "맑고 깨끗한 동양적 무쌍 미인", "vibe": "자연스럽고 맑은 미소, 편안하고 세련된 눈매"},
    {"name": "박보영", "query": "박보영 (배우)", "file": "park_bo_young.jpg", "gender": "female", "category": "배우", "face_type": "원조 국민 여동생 러블리 강아지상", "vibe": "동그란 눈망울, 작고 귀여운 하관과 사랑스러운 미소"},
    {"name": "김유정", "query": "김유정 (배우)", "file": "kim_yoo_jung.jpg", "gender": "female", "category": "배우", "face_type": "싱그럽고 사랑스러운 과즙상", "vibe": "크고 반짝이는 눈망울과 환한 반달 눈웃음"},
    {"name": "김다미", "query": "김다미 (배우)", "file": "kim_da_mi.jpg", "gender": "female", "category": "배우", "face_type": "신비로운 무쌍 베이비페이스", "vibe": "순수하면서도 묘한 매력을 풍기는 맑은 눈매"},
    {"name": "신민아", "query": "신민아", "file": "shin_min_a.jpg", "gender": "female", "category": "배우", "face_type": "사랑스러운 보조개 베이비페이스", "vibe": "쏙 들어가는 보조개, 도톰한 입술과 싱그러운 눈매"},
    {"name": "박소담", "query": "Park So-dam", "lang": "en", "file": "park_so_dam.jpg", "gender": "female", "category": "배우", "face_type": "단아하고 세련된 동양미 무쌍", "vibe": "정갈한 선과 맑은 눈빛, 개성 있는 매력"},
    {"name": "츄", "query": "Chuu", "lang": "en", "file": "chuu.jpg", "gender": "female", "category": "가수", "face_type": "비타민 같은 하트 미소 러블리상", "vibe": "보는 순간 미소 짓게 만드는 환한 눈웃음과 긍정 에너지"},
    {"name": "정호연", "query": "Hoyeon Jung", "lang": "en", "file": "jung_ho_yeon.jpg", "gender": "female", "category": "배우", "face_type": "도톰한 입술과 매혹적인 무쌍 모델상", "vibe": "동양적이고 이국적인 매력, 트렌디한 마스크"},
    {"name": "화사", "query": "화사", "file": "hwasa.jpg", "gender": "female", "category": "가수", "face_type": "이국적이고 당당한 걸크러시 마스크", "vibe": "매혹적인 아치형 눈썹과 매력적인 입술선, 독보적인 아우라"},
    {"name": "이효리", "query": "이효리", "file": "lee_hyo_ri.jpg", "gender": "female", "category": "가수", "face_type": "반달 눈웃음의 원조 걸크러시 퀸", "vibe": "보는 이를 무장해제시키는 털털하고 사랑스러운 반달 눈웃음"},
    {"name": "엄정화", "query": "엄정화", "file": "uhm_jung_hwa.jpg", "gender": "female", "category": "배우", "face_type": "도회적이고 매혹적인 글래머러스상", "vibe": "크고 깊은 눈망울과 당당하고 우아한 매력"},
    {"name": "장도연", "query": "장도연", "file": "jang_do_yeon.jpg", "gender": "female", "category": "방송인", "face_type": "세련된 숏컷과 시원시원한 모델상", "vibe": "지적이고 단정한 이목구비, 훤칠하고 호감 넘치는 마스크"},
    {"name": "박나래", "query": "박나래", "file": "park_na_rae.jpg", "gender": "female", "category": "방송인", "face_type": "개성 넘치고 유쾌한 에너지의 호감형", "vibe": "동글동글한 눈매와 친근하고 에너지 넘치는 인상"}
]

WIKIDATA_QUERY = """
SELECT ?p ?ko ?en ?gender ?cat ?img ?links ?occ WHERE {
  ?p wdt:P27 wd:Q884; wdt:P31 wd:Q5; wdt:P21 ?gender; wdt:P373 ?cat; wikibase:sitelinks ?links.
  ?p wdt:P106 ?occ. VALUES ?occ { wd:Q33999 wd:Q10800557 wd:Q10798782 wd:Q177220 wd:Q947873 wd:Q4610556 wd:Q245068 wd:Q488205 }
  ?p wdt:P569 ?birth. FILTER(YEAR(?birth) >= 1950)
  FILTER NOT EXISTS { ?p wdt:P570 ?death }
  OPTIONAL { ?p wdt:P18 ?img }
  ?p rdfs:label ?ko FILTER(LANG(?ko)="ko")
  OPTIONAL { ?p rdfs:label ?en FILTER(LANG(?en)="en") }
} ORDER BY DESC(?links) LIMIT 4000
"""

_print_lock = threading.Lock()

# Wikimedia returns 429 when bots go too fast: one shared request clock for all worker threads,
# and a 429's Retry-After pauses everyone.
MIN_REQUEST_INTERVAL = 0.35
_rate_lock = threading.Lock()
_next_request_at = 0.0


def log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def _throttle():
    global _next_request_at
    with _rate_lock:
        now = time.monotonic()
        wait = _next_request_at - now
        _next_request_at = max(now, _next_request_at) + MIN_REQUEST_INTERVAL
    if wait > 0:
        time.sleep(wait)


def _backoff(error: Exception, attempt: int):
    global _next_request_at
    if getattr(error, "code", None) == 429:
        try:
            delay = float(error.headers.get("Retry-After") or 10) + 1
        except (TypeError, ValueError, AttributeError):
            delay = 11
        with _rate_lock:
            _next_request_at = max(_next_request_at, time.monotonic() + delay)
    else:
        delay = 2 * (attempt + 1)
    time.sleep(delay)


def _fetch(url: str, timeout: float, accept_json: bool, retries: int = 6) -> bytes:
    headers = {**HTTP_HEADERS, "Accept": "application/json"} if accept_json else HTTP_HEADERS
    for attempt in range(retries):
        _throttle()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1 or getattr(e, "code", None) == 404:
                raise
            _backoff(e, attempt)


def http_json(url: str, timeout: float = 20):
    return json.loads(_fetch(url, timeout, accept_json=True).decode("utf-8"))


def http_bytes(url: str, timeout: float = 20) -> bytes:
    return _fetch(url, timeout, accept_json=False)


def clean_name(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", name).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def fetch_wikidata_people(limit: int) -> dict:
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": WIKIDATA_QUERY, "format": "json"})
    rows = http_json(url, timeout=90)["results"]["bindings"]
    people = {}
    for b in rows:
        qid = b["p"]["value"].rsplit("/", 1)[-1]
        person = people.setdefault(qid, {
            "qid": qid,
            "name": clean_name(b["ko"]["value"]),
            "name_en": clean_name(b.get("en", {}).get("value", "")),
            "gender": GENDER_QIDS.get(b["gender"]["value"].rsplit("/", 1)[-1]),
            "commons_category": b["cat"]["value"],
            "main_image": b.get("img", {}).get("value", "").rsplit("/", 1)[-1],
            "sitelinks": int(b["links"]["value"]),
            "occupations": set(),
        })
        person["occupations"].add(b["occ"]["value"].rsplit("/", 1)[-1])

    for person in people.values():
        occs = person.pop("occupations")
        person["category"] = next((c for q, c in OCCUPATION_CATEGORY if occs & q), "연예인")

    ranked = sorted((p for p in people.values() if p["gender"]), key=lambda p: -p["sitelinks"])
    return {p["qid"]: p for p in ranked[:limit]}


def resolve_seed(seed: dict):
    """Find the Wikidata item for a hand-written seed via Korean (or English) Wikipedia search."""
    for lang, query in [(seed.get("lang", "ko"), seed["query"]), ("en", seed.get("en_query"))]:
        if not query:
            continue
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
            "prop": "pageprops", "ppprop": "wikibase_item", "format": "json"})
        pages = http_json(url).get("query", {}).get("pages", {})
        qid = next((p.get("pageprops", {}).get("wikibase_item") for p in pages.values()), None)
        if not qid:
            continue
        ent = http_json("https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode({
            "action": "wbgetentities", "ids": qid, "props": "claims|labels|aliases", "languages": "ko|en",
            "format": "json"}))["entities"][qid]
        # Search can land on a different person (e.g. a namesake); require the Korean label/alias to match
        ko_names = [ent.get("labels", {}).get("ko", {}).get("value", "")]
        ko_names += [a["value"] for a in ent.get("aliases", {}).get("ko", [])]
        if seed["name"] not in {clean_name(n) for n in ko_names}:
            log(f"[WARN] seed '{seed['name']}': search for '{query}' found {qid} ({ko_names[0] or '?'}), skipped")
            continue
        claims = ent.get("claims", {})

        def claim(pid):
            try:
                return claims[pid][0]["mainsnak"]["datavalue"]["value"]
            except (KeyError, IndexError):
                return None

        if not claim("P373") and not claim("P18"):
            # Namesakes without any Commons photo are almost never the celebrity we mean
            log(f"[WARN] seed '{seed['name']}': {qid} has no Commons photos, trying next query")
            continue
        return {
            "qid": qid,
            "name_en": clean_name(ent.get("labels", {}).get("en", {}).get("value", "")),
            "commons_category": claim("P373"),
            "main_image": claim("P18") or "",
        }
    return None


def list_commons_photos(person: dict) -> list:
    """(file title, thumb url) pairs: main Wikidata image first, then deepcat search results."""
    titles = []
    if person.get("main_image"):
        titles.append("File:" + urllib.parse.unquote(person["main_image"]))
    if person.get("commons_category"):
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srnamespace": 6, "srlimit": 25,
            "srsearch": f'deepcat:"{person["commons_category"]}" filetype:bitmap', "format": "json"})
        try:
            titles += [r["title"] for r in http_json(url).get("query", {}).get("search", [])]
        except Exception as e:
            log(f"[WARN] Commons search failed for {person['name']}: {e}")
    titles = list(dict.fromkeys(t for t in titles if not re.search(r"(?i)logo|signature|autograph|\.svg$", t)))
    titles = titles[:PHOTOS_PER_PERSON + 4]
    if not titles:
        return []

    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
        "iiprop": "url|mime", "iiurlwidth": THUMB_WIDTH, "format": "json"})
    data = http_json(url)
    # The API normalizes titles (e.g. underscores); map back to what we asked for
    normalized = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
    info = {}
    for page in data.get("query", {}).get("pages", {}).values():
        ii = (page.get("imageinfo") or [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png", "image/webp"):
            info[normalized.get(page["title"], page["title"])] = ii.get("thumburl") or ii.get("url")
    return [(t, info[t]) for t in titles if t in info][:PHOTOS_PER_PERSON]


def load_person_photos(person: dict) -> list:
    """Download (or read cached) photos; returns [(file title, raw bytes)]."""
    folder = os.path.join(CACHE_DIR, person["qid"])
    manifest_path = os.path.join(folder, "manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            cached = json.load(f)
        # Reuse only complete downloads (older runs lost photos to rate limiting)
        if isinstance(cached, dict) and len(cached["photos"]) >= cached["expected"]:
            manifest = cached["photos"]
    if manifest is None:
        os.makedirs(folder, exist_ok=True)
        listed = list_commons_photos(person)
        manifest = []
        for i, (title, url) in enumerate(listed):
            try:
                data = http_bytes(url)
            except Exception as e:
                log(f"[WARN] download failed {title}: {e}")
                continue
            fname = f"{i:02d}.img"
            with open(os.path.join(folder, fname), "wb") as f:
                f.write(data)
            manifest.append({"title": title, "file": fname})
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"expected": len(listed), "photos": manifest}, f, ensure_ascii=False)

    photos = []
    for m in manifest:
        path = os.path.join(folder, m["file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                photos.append((m["title"], f.read()))
    return photos


def consensus_embedding(photos: list):
    """
    Identify the person as the face that recurs across their photos.
    Returns (mean embedding, n_matched, (title, img_bgr, face_row) for display) or None.
    """
    faces = []  # (photo index, title, img, face_row, embedding)
    for idx, (title, data) in enumerate(photos):
        img = decode_image_bgr(data)
        if img is None:
            continue
        for row in detect_faces_yunet(img, score_threshold=0.7):
            if min(row[2], row[3]) < MIN_FACE_PX:
                continue
            emb = embed_face(img, row)
            if emb is not None:
                faces.append((idx, title, img, row, emb))
    if not faces:
        return None

    embs = np.array([f[4] for f in faces])
    sims = embs @ embs.T
    photo_ids = np.array([f[0] for f in faces])

    # Support = number of *other* photos containing a face matching this one
    supports = []
    for i in range(len(faces)):
        matches = np.where(sims[i] >= SAME_PERSON_COS)[0]
        supports.append(len({photo_ids[j] for j in matches if photo_ids[j] != photo_ids[i]}))

    if max(supports) > 0:
        anchor = max(range(len(faces)), key=lambda i: (supports[i], faces[i][3][2] * faces[i][3][3]))
    else:
        # No recurring face: trust only a solo face in the Wikidata main image (photo 0)
        solo_main = [i for i in range(len(faces)) if photo_ids[i] == 0 and (photo_ids == 0).sum() == 1]
        if not solo_main:
            return None
        anchor = solo_main[0]

    # Best matching face per photo
    chosen = {}
    for i, f in enumerate(faces):
        if i == anchor or sims[anchor, i] >= SAME_PERSON_COS:
            if f[0] not in chosen or sims[anchor, i] > sims[anchor, chosen[f[0]]]:
                chosen[f[0]] = i
    idxs = list(chosen.values())
    mean = embs[idxs].mean(axis=0)
    mean /= np.linalg.norm(mean)

    # Display photo: large face that is most typical of the person
    best = max(idxs, key=lambda i: float(embs[i] @ mean) + min(faces[i][3][2], 200) / 400.0)
    return mean, len(idxs), (faces[best][1], faces[best][2], faces[best][3])


def save_display_photo(img_bgr, face_row, out_path: str):
    h, w = img_bgr.shape[:2]
    fx, fy, fw, fh = map(int, face_row[:4])
    px, py, size, _ = compute_square_face_box(fx, fy, fw, fh, w, h, [float(v) for v in face_row[4:8]])
    crop = np.ascontiguousarray(img_bgr[py:py + size, px:px + size][:, :, ::-1])
    Image.fromarray(crop).resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.LANCZOS).save(
        out_path, format="JPEG", quality=85, optimize=True)


def build_person(person: dict):
    photos = load_person_photos(person)
    if not photos:
        return None
    result = consensus_embedding(photos)
    if result is None:
        return None
    mean, n_matched, (title, img, row) = result

    filename = person["file"]
    save_display_photo(img, row, os.path.join(CELEB_DIR, filename))
    entry = {
        "name": person["name"],
        "name_en": person.get("name_en", ""),
        "qid": person["qid"],
        "gender": person["gender"],
        "category": person["category"],
        "filename": filename,
        "photo_url": f"/facematching/static/celebrities/{filename}",
        "photo_source": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
        "n_photos": n_matched,
        "embedding": [round(float(v), 5) for v in mean],
    }
    for key in ("face_type", "vibe"):
        if person.get(key):
            entry[key] = person[key]
    return entry


def collect_candidates(limit: int) -> list:
    log("Querying Wikidata...")
    people = fetch_wikidata_people(limit)
    log(f"  {len(people)} people from Wikidata")

    for seed in CELEBRITY_CATALOG:
        try:
            info = resolve_seed(seed)
        except Exception as e:
            log(f"[WARN] seed lookup failed for {seed['name']}: {e}")
            info = None
        if not info:
            log(f"[WARN] seed not found on Wikidata: {seed['name']}")
            continue
        base = people.get(info["qid"], {**info, "sitelinks": 0})
        base.update({
            "name": seed["name"], "gender": seed["gender"], "category": seed["category"],
            "file": seed["file"], "face_type": seed["face_type"], "vibe": seed["vibe"],
            "commons_category": base.get("commons_category") or info["commons_category"],
            "main_image": base.get("main_image") or info["main_image"],
        })
        base.setdefault("name_en", info["name_en"])
        people[info["qid"]] = base
        time.sleep(0.1)
    return list(people.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600, help="max people taken from Wikidata")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(YUNET_PATH) or not os.path.exists(ARCFACE_PATH):
        raise SystemExit("[ERROR] models/face_detection_yunet.onnx and models/arcface_w600k_r50.onnx are required")

    candidates = collect_candidates(args.limit)

    # Assign unique display filenames up front (seeds keep theirs; English slugs can collide)
    used = {p["file"] for p in candidates if p.get("file")}
    for p in candidates:
        if not p.get("file"):
            slug = slugify(p.get("name_en") or p.get("commons_category") or "") or p["qid"].lower()
            p["file"] = f"{slug}.jpg" if f"{slug}.jpg" not in used else f"{slug}_{p['qid'].lower()}.jpg"
            used.add(p["file"])

    log(f"Building embeddings for {len(candidates)} candidates...")

    entries, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_person, p): p for p in candidates}
        for n, fut in enumerate(as_completed(futures), start=1):
            person = futures[fut]
            try:
                entry = fut.result()
            except Exception as e:
                log(f"[ERROR] {person['name']}: {e}")
                entry = None
            if entry:
                entries.append(entry)
            else:
                failed.append(person["name"])
            if n % 25 == 0:
                log(f"  {n}/{len(candidates)} processed, {len(entries)} ok")

    entries.sort(key=lambda e: e["name"])
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    log(f"\n[DONE] {len(entries)} celebrities saved to {DB_PATH} ({len(failed)} skipped)")
    if failed:
        log("Skipped: " + ", ".join(failed))


if __name__ == "__main__":
    main()
