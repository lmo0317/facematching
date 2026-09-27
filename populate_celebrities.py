import os
import urllib.request
import urllib.parse
import json
import time
import io
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(BASE_DIR, "static", "celebrities")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "celebrity_db.json")

os.makedirs(CELEB_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "FaceMatchCelebrityLibrary/2.0 (https://minohlee.mooo.com; admin@minohlee.mooo.com)"
}

# 100 Representative Korean Celebrities across diverse facial structures
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
    {"name": "이도현", "query": "이도현 (배우)", "file": "lee_do_hyun.jpg", "gender": "male", "category": "배우", "face_type": "매력적인 입꼬리와 맑고 깊은 눈", "vibe": "청량한 소년미와 성숙한 카리스마가 공존하는 마스크"},

    # 듬직 / 선 굵은 카리스마 / 상남자형
    {"name": "이정재", "query": "이정재", "file": "lee_jung_jae.jpg", "gender": "male", "category": "배우", "face_type": "매력적인 광대와 기품 있는 미소", "vibe": "고급스럽고 중후한 매력, 웃을 때 번지는 눈웃음과 턱선"},
    {"name": "하정우", "query": "하정우", "file": "ha_jung_woo.jpg", "gender": "male", "category": "배우", "face_type": "선 굵고 묵직한 카리스마", "vibe": "남성미 넘치는 턱선과 짙은 눈빛, 거침없는 아우라"},
    {"name": "이병헌", "query": "이병헌", "file": "lee_byung_hun.jpg", "gender": "male", "category": "배우", "face_type": "강렬하고 묵직한 하관과 깊은 눈빛", "vibe": "단단한 턱선, 압도적인 눈빛과 신뢰감을 주는 골격"},
    {"name": "마동석", "query": "마동석", "file": "ma_dong_seok.jpg", "gender": "male", "category": "배우", "face_type": "다부지고 듬직한 베어(곰)상", "vibe": "묵직하고 단단한 하관과 푸근하면서도 압도적인 카리스마"},
    {"name": "황정민", "query": "황정민 (배우)", "file": "hwang_jung_min.jpg", "gender": "male", "category": "배우", "face_type": "사람 냄새 나는 따뜻하고 깊은 마스크", "vibe": "진솔한 눈매와 친근한 주름, 진정성 넘치는 인상"},
    {"name": "조진웅", "query": "조진웅", "file": "cho_jin_woong.jpg", "gender": "male", "category": "배우", "face_type": "듬직하고 웅장한 호랑이상", "vibe": "묵직한 안면 골격과 신뢰감을 주는 중후한 매력"},

    # 친근 / 호감 / 푸근 / 동글형
    {"name": "안재홍", "query": "안재홍 (배우)", "file": "ahn_jae_hong.jpg", "gender": "male", "category": "배우", "face_type": "순박하고 정감 가는 순둥이 호감형", "vibe": "동글동글한 콧망울과 온화한 눈매, 보는 이를 편안하게 하는 인상"},
    {"name": "조세호", "query": "조세호", "file": "cho_sae_ho.jpg", "gender": "male", "category": "방송인", "face_type": "동글동글 푸근한 복덩이 미소", "vibe": "통통하고 귀여운 볼살, 친근하고 유쾌한 에너지"},
    {"name": "싸이", "query": "싸이 (가수)", "file": "psy.jpg", "gender": "male", "category": "가수", "face_type": "에너지 넘치고 개성 있는 둥근 얼굴", "vibe": "자신감 넘치는 눈빛과 시원한 미소, 독보적인 캐릭터"},
    {"name": "유재석", "query": "유재석", "file": "yoo_jae_suk.jpg", "gender": "male", "category": "방송인", "face_type": "친근하고 밝은 국민 MC상", "vibe": "선하고 지적인 눈매, 시원한 입매와 신뢰감을 주는 표정"},
    {"name": "신동엽", "query": "신동엽 (방송인)", "file": "shin_dong_yup.jpg", "gender": "male", "category": "방송인", "face_type": "오밀조밀 모인 개성 만점 호감상", "vibe": "장난기 넘치는 눈웃음과 재치 있는 인상"},
    {"name": "기안84", "query": "기안84", "file": "kian84.jpg", "gender": "male", "category": "방송인", "face_type": "날것 그대로의 순수하고 호탕한 인상", "vibe": "꾸밈없는 담백한 눈매와 시원털털한 미소"},
    {"name": "손흥민", "query": "손흥민", "file": "son_heung_min.jpg", "gender": "male", "category": "스포츠", "face_type": "기분 좋아지는 반달 눈웃음의 승부사", "vibe": "동양적인 매력의 무쌍 눈매와 건강하고 활기찬 에너지"},
    {"name": "임영웅", "query": "임영웅", "file": "lim_young_woong.jpg", "gender": "male", "category": "가수", "face_type": "따뜻하고 훈훈한 국민 힐러상", "vibe": "선한 눈망울과 차분한 입매, 신뢰와 위로를 주는 인상"},
    {"name": "황정민", "query": "황정민", "file": "hwang_jung_min.jpg", "gender": "male", "category": "배우", "face_type": "사람 냄새 나는 깊은 마스크", "vibe": "진솔한 눈매와 친근한 주름, 진정성 넘치는 인상"},
    {"name": "이도현", "query": "Lee Do-hyun", "lang": "en", "file": "lee_do_hyun.jpg", "gender": "male", "category": "배우", "face_type": "매력적인 입꼬리와 맑고 깊은 눈", "vibe": "청량한 소년미와 성숙한 카리스마가 공존하는 마스크"},
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

def fetch_image_if_missing(query_name, filename, lang="ko"):
    out_path = os.path.join(CELEB_DIR, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 4000:
        return True

    print(f"Fetching {query_name} ({lang})...")
    encoded = urllib.parse.quote(query_name)
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={encoded}&gsrlimit=1&prop=pageimages&piprop=original&format=json"

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode('utf-8'))
            pages = data.get('query', {}).get('pages', {})
            img_url = None
            for pid, info in pages.items():
                if 'original' in info and 'source' in info['original']:
                    img_url = info['original']['source']
                    break

            if not img_url:
                print(f"[SKIP] No image found for {query_name}")
                return False

            img_req = urllib.request.Request(img_url, headers=HEADERS)
            with urllib.request.urlopen(img_req, timeout=12) as ir:
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

                cropped = img.crop((left, top, right, bottom))
                resized = cropped.resize((600, 600), Image.Resampling.LANCZOS)
                resized.save(out_path, format="JPEG", quality=85, optimize=True)

            print(f"[DOWNLOADED] {filename} ({round(os.path.getsize(out_path)/1024)} KB)")
            return True
    except Exception as e:
        print(f"[ERROR] {query_name}: {e}")
        return False


def build_celebrity_embeddings_database():
    """Extract YuNet faces and SFace 128-D embeddings for all celebrities and save DB."""
    yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
    sface_path = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

    if not os.path.exists(sface_path) or not os.path.exists(yunet_path):
        print(f"[ERROR] Model files missing in {MODELS_DIR}")
        return

    recognizer = cv2.FaceRecognizerSF.create(sface_path, '')
    print("Building SFace embeddings database...")

    db_entries = []
    success_count = 0

    for item in CELEBRITY_CATALOG:
        fname = item["file"]
        img_path = os.path.join(CELEB_DIR, fname)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(yunet_path, '', (w, h), 0.45, 0.3, 5)
        _, faces = det.detect(img)
        if faces is None or len(faces) == 0:
            print(f"[WARN] No face found in {fname}")
            continue

        # Choose largest face if multiple
        best_face = max(faces, key=lambda f: f[2] * f[3])
        aligned = recognizer.alignCrop(img, best_face)
        feat = recognizer.feature(aligned)
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            feat_norm = (feat / norm).flatten().tolist()
        else:
            continue

        entry = {
            "name": item["name"],
            "gender": item["gender"],
            "category": item["category"],
            "filename": fname,
            "face_type": item["face_type"],
            "vibe": item["vibe"],
            "photo_url": f"/facematching/static/celebrities/{fname}",
            "embedding": feat_norm
        }
        db_entries.append(entry)
        success_count += 1

    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db_entries, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Built celebrity database with {success_count} entries saved to {DB_PATH}")


if __name__ == "__main__":
    print("Step 1: Downloading missing images from Wikipedia...")
    for item in CELEBRITY_CATALOG:
        fetch_image_if_missing(item["query"], item["file"], item.get("lang", "ko"))
        time.sleep(0.15)

    print("\nStep 2: Generating Deep Face Embeddings (YuNet + SFace)...")
    build_celebrity_embeddings_database()
