# AGENTS.md — FaceMatch AI 프로젝트 지침

이 문서는 이 저장소에서 작업할 때 따라야 할 구조·규칙·주의사항을 정리한 개발 지침입니다.
사용자용 소개/접속 정보는 [README.md](README.md)를 참고하세요.

---

## 1. 프로젝트 개요

얼굴 인식 임베딩(ArcFace) + Gemma 4 E4B 멀티모달 LLM을 이용한 **얼굴 닮음 분석 웹앱**입니다.
핵심 기능은 두 가지입니다.

| 탭 | 기능 | API |
|---|---|---|
| `compare` | **사진 2장 닮음 분석** (가족·붕어빵 지수) | `POST /api/compare-json` |
| `celeb` | **내 사진으로 닮은 연예인 찾기** (TOP 5 + 실사진, 사진 최대 3장) | `POST /api/find-celebrity` |

**역할 분담 원칙 (중요)**
- **점수·후보 선정은 ArcFace 임베딩이 담당**합니다. 같은 입력이면 항상 같은 결과가 나오고 실제 얼굴 구조에 근거합니다.
- **Gemma 4는 설명 문장과 부위별 패턴만 담당**합니다. Gemma 4 E4B는 얼굴로 연예인을 식별하지 못하고(본인 사진을 넣어도 인기 연예인 이름을 돌려막기), 가족/남남 점수도 구분하지 못하는 것이 실측으로 확인됐습니다. Gemma에게 점수나 후보 선정을 다시 맡기지 마세요.

- 백엔드: Python 3.12, FastAPI + Uvicorn (단일 프로세스, 빌드 단계 없음)
- 프론트엔드: 순수 HTML/JS (프레임워크·번들러 없음), Tailwind/Lucide/Cropper.js는 CDN 로드
- LLM: 로컬 `llama-server`(llama.cpp)의 OpenAI 호환 `/v1/chat/completions` 엔드포인트
- 얼굴 검출: OpenCV YuNet / 얼굴 임베딩: InsightFace ArcFace `w600k_r50` (512-D, onnxruntime CPU)

---

## 2. 디렉터리 구조

```
app.py                     # FastAPI 앱: 라우트, 얼굴 검출 API, 2장 비교(ArcFace+Gemma 결합)
celebrity_service.py       # 닮은 연예인 탐색 (ArcFace 최근접 검색 + Gemma 설명), app.py에서 지연 import
face_utils.py              # 공용: 이미지 정규화, YuNet 검출, ArcFace 정렬·임베딩, 유사도→% 환산, LLM JSON 추출
populate_celebrities.py    # 오프라인: Wikidata/Commons에서 연예인 사진 수집 → 다중 사진 평균 임베딩 DB 생성
tools/calibrate_scores.py  # 오프라인: 가족/남남/동일인 코사인 분포 측정 (유사도→% 환산 기준 근거)
tools/eval_lookalike.py    # 오프라인: 닮은 연예인 검색 정확도 평가 (사람들이 닮았다고 말하는 쌍 기준)
tools/lookalike_pairs.txt  # 평가용 닮은꼴 연예인 쌍 (나무위키·기사 출처)
celebrity_db.json          # 연예인 메타데이터 + 512-D 평균 임베딩 (populate 산출물, 한 줄에 1명)
celebrity_visual.npz       # 연예인 대표 사진의 CLIP 임베딩 (시각 재정렬용, populate 산출물)
models/
  face_detection_yunet.onnx    # 얼굴 검출 (git 추적 — .gitignore 예외)
  arcface_w600k_r50.onnx       # 얼굴 임베딩 (git 미추적, 서버에 수동 배치)
  clip_vit_b32_vision_q.onnx   # CLIP 이미지 인코더(양자화, 약 90MB) — 시각 재정렬용 (git 미추적, 서버에 수동 배치)
data/                      # 빌드 캐시 (git 미추적): celeb_cache/<QID>/, kin_cache/, build.log
static/
  index.html               # 단일 페이지 UI (탭 2개 + 크롭/웹캠 모달)
  js/app.js                # 전역 상태 + 모든 UI 로직
  css/style.css            # Tailwind 보조 커스텀 스타일
  samples/                 # 비교 탭 예제 프리셋 이미지
  celebrities/             # 연예인 대표 사진 (얼굴 중심 360x360 JPEG, populate 산출물)
facematch.service          # systemd user service 정의
requirements.txt
```

---

## 3. 실행 & 환경 변수

```bash
pip install -r requirements.txt
python app.py            # 또는: uvicorn app:app --host 0.0.0.0 --port 8501
```

ArcFace 모델은 저장소에 없으므로 처음 한 번 받아야 합니다 (InsightFace `buffalo_l` 패키지 안의 `w600k_r50.onnx`):
```bash
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
python -c "import zipfile; zipfile.ZipFile('/tmp/buffalo_l.zip').extract('w600k_r50.onnx', 'models')"
mv models/w600k_r50.onnx models/arcface_w600k_r50.onnx
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | Gemma 4 llama-server 주소 |
| `MODEL_NAME` | `gemma-4-e4b-it-q4km` | 요청 payload의 model 이름 |
| `HOST` / `PORT` | `0.0.0.0` / `8501` | `python app.py` 실행 시에만 사용 |

- 로컬(Windows)에는 LLM 서버가 없습니다. 두 기능 모두 LLM 없이도 동작하며(ArcFace 점수 + 기본 문구), 설명 문장은 112 서버에서만 생성됩니다.
- LLM을 로컬에서 쓰려면 SSH 터널로 112 서버의 8081을 포워딩한 뒤 `LLAMA_SERVER_URL`을 지정하세요.
- 자동화 테스트는 없습니다. 닮은 연예인 검색을 바꿀 때는 `python tools/eval_lookalike.py`로 R@3/R@10/중앙 순위를 변경 전후 비교하세요(얼굴 뱅크 캐시가 없으면 몇 분 걸림). 2장 비교는 예제 프리셋으로 확인합니다.

---

## 4. 백엔드 아키텍처

### 4.1 라우팅 규칙 (중요)
앱은 **루트(`/`)와 리버스 프록시 하위 경로(`/facematching/`) 양쪽에서 동작**해야 합니다.
- 모든 라우트는 데코레이터를 **2개씩** 붙입니다: `@app.post("/api/x")` + `@app.post("/facematching/api/x")`
- 정적 파일도 `/static`과 `/facematching/static` 두 곳에 마운트됩니다.
- **새 엔드포인트를 추가할 때 반드시 두 경로 모두 등록**하세요.

| 메서드 | 경로 | 역할 |
|---|---|---|
| GET | `/`, `/facematching/` | `static/index.html` 서빙 |
| GET | `/api/health` | llama-server `/v1/models` 연결 상태 확인 |
| GET | `/api/samples` | 비교 탭 예제 프리셋 목록 (하드코딩) |
| POST | `/api/detect-faces` | YuNet(→ Haar 폴백) 얼굴 검출, 1:1 크롭 박스 + 80px 썸네일 반환 |
| POST | `/api/compare` | multipart 업로드 비교 (프론트 미사용, 외부 호출용) |
| POST | `/api/compare-json` | base64 JSON 비교 (프론트가 사용) |
| POST | `/api/find-celebrity` | 닮은 연예인 TOP 5. `extra_images_base64`(같은 사람 추가 사진 최대 2장) 선택. `describe:false`면 설명 없이 순위만 약 2초에 반환(`descriptions_pending`). 얼굴 미검출 시 400 |
| POST | `/api/describe-celebrity` | 2단계: `qids`·`percents`로 이미 선정된 후보의 Gemma 설명. `part`: `top`(내 얼굴 특징 + 1위 상세, 약 7초) / `others`(2~5위 간결, 약 31초) / `all` |

### 4.2 얼굴 임베딩 (`face_utils`)
- `decode_image_bgr`(EXIF 회전 반영) → `detect_faces_yunet`(긴 변 1280px로 축소 검출 후 원본 좌표로 복원) → 가장 큰 얼굴
- `align_face`: YuNet 5점 랜드마크를 ArcFace 표준 템플릿(112x112)에 similarity transform으로 정렬. YuNet 랜드마크 순서(화면 왼쪽 눈 먼저)는 템플릿 순서와 같습니다.
- `embed_face`: RGB, (x-127.5)/127.5 정규화 → 512-D → L2 정규화. 코사인 = 내적.
- **임베딩 모델을 바꾸면 `celebrity_db.json`을 반드시 재생성**해야 합니다. 로드 시 차원이 다른 항목은 버리고 에러 로그를 남깁니다.

### 4.3 유사도 → % 환산 (`similarity_to_percent`)
- `SIMILARITY_ANCHORS`의 프로필별 구간 선형 보간: `family`(2장 닮음), `identical`(동일인 판정), `celebrity`(연예인 검색)
- 기준값은 `tools/calibrate_scores.py`로 측정한 실제 분포(가족 쌍 / 남남 쌍 / 같은 사람의 다른 사진)에서 정했습니다. **임의로 바꾸지 말고, 바꿀 때는 스크립트를 다시 돌려 근거를 확인**하세요.
- `adjust_part_scores`: LLM 부위별 점수의 **상대 패턴만** 유지하고 수준은 최종 점수에 맞춰 재중심화합니다.

### 4.4 사진 2장 비교 (`execute_face_comparison`)
1. 두 사진의 ArcFace 코사인 → `similarity_to_percent` (모드 `identical`이면 identical 프로필, 그 외 family)
2. Gemma 4 비교 요청(`request_gemma_comparison`). 측정값을 프롬프트에 참고로 넣어 소견 문장이 점수와 모순되지 않게 합니다. Gemma 원점수는 기존 `calibrate_similarity_score`로 보정.
3. 최종 점수 = `FACE_SCORE_WEIGHT[mode] × ArcFace% + 나머지 × Gemma%` (family/celebrity 0.8, identical 1.0)
4. 응답 `data.score_breakdown`에 측정값·코사인·LLM 점수·가중치를 담아 프론트가 표시합니다.
5. **Gemma 장애 시에도 ArcFace만으로 결과를 반환**합니다. 얼굴을 못 찾으면 Gemma 점수만 사용하고 그 사실을 `environmental_factors`에 적습니다. 둘 다 불가하면 기존 에러 매핑(502/503/504).

### 4.5 닮은 연예인 (`celebrity_service.execute_celebrity_lookalike`)
1. 사용자 ArcFace 임베딩 vs DB 전체 평균 임베딩 코사인 → 순위. 추가 사진이 있으면 메인 사진과 코사인 ≥ 0.25인 것만 평균에 넣고 나머지는 다른 사람으로 보고 제외(`photos_used`/`photos_rejected`)
2. 성별: 필터 `male`/`female`은 코드에서 강제. `auto`는 최근접 15명의 성별을 유사도 가중 다수결로 추정.
3. **시각 재정렬**: 상위 `VISUAL_POOL_K`(10)명 안에서 `ArcFace 코사인 + VISUAL_WEIGHT(0.03) × z(CLIP 유사도)`로 순서를 보정합니다. CLIP은 사용자 사진의 머리 포함 크롭 vs 연예인 대표 사진(`celebrity_visual.npz`). 상위 후보는 사실상 동점(1·2위 차이 중앙값 0.017)이라 ArcFace 순서만으로는 화면에 보이는 1위가 TOP 5 중 눈으로 가장 닮은 경우가 23%(무작위 수준)였고, 보정 후 40%입니다(평가 세트 정확도는 동일). 가중치를 올리면 이 비율은 오르지만 0.05부터 정확도가 떨어집니다.
4. TOP 5를 보정 점수로 `celebrity` 프로필 % 환산. **후보와 점수는 여기서 확정**됩니다. 검색마다 `celebrity search: ...` 로그(성별·사진 수·TOP 5 이름/점수/코사인, 사진은 저장 안 함)를 남깁니다.
5. Gemma에는 사용자 사진 + TOP 5(`GEMMA_DESCRIBED`) 사진을 주고 설명만 요청합니다(이름 변경 금지): 내 얼굴 특징(인상 키워드·얼굴형·눈매·코·입·전체 인상), 후보별 요약·닮은 이유(1위 3문장)·부위별 비교(`part_notes`)·다른 점(`differences`)·부위별 점수. 실패 시 DB의 `face_type`/`vibe`나 기본 문구로 대체.
   - 한 번에 쓰면 50~70초가 걸려 **공개 경로(nginx `location /` → Service Hub:3000 → 8501)의 기본 60초 제한에 걸려 멈췄습니다.** 그래서 나눕니다: 프론트는 `describe:false`로 순위를 먼저 그리고(`renderCelebrityResults`), `fetchCelebDescriptions`가 `part=top`(내 얼굴 특징 + 1위 상세)만 받아 텍스트를 합칩니다. **2~5위는 AI 설명 없이 사진·이름·분야·점수만** 보여 줍니다(느리다는 피드백으로 `part=others` 호출을 뺐음, 백엔드에는 남아 있음)(요청당 80초 클라이언트 타임아웃, 실패 시 "다시 시도" 버튼이 빠진 부분만 재요청). 새 검색이 시작되면 `celebSearchToken`으로 늦게 온 응답을 버립니다. 서버 로그 `gemma describe mode=... took Ns`로 소요 시간을 확인할 수 있습니다.
   - 설명 요청 하나가 60초를 넘기지 않게 유지하세요(출력 길이·후보 수를 늘리면 다시 나눠야 합니다).
   - `top` 속도 근거(실측): 이미지 2장 prefill 약 1초(mmproj GPU) + 출력 약 420토큰 × 13ms ≈ 5.5초. 그래서 `top`은 `COMPACT_*_SCHEMA`(항목별 글자 수 제한, 한 줄 JSON)로 출력을 줄입니다. 설명을 길게 늘리면 그만큼 느려집니다.
6. DB에 없는 사람은 절대 결과에 나오지 않습니다(실시간 위키 검색·대체 사진 없음).

### 4.6 얼굴 검출 API (`detect_faces_in_image`)
- EXIF 회전 보정 후 YuNet(score 0.55) → 실패 시 Haar Cascade 폴백
- 얼굴은 x좌표(왼→오) 순 정렬, 라벨은 `face_position_label`(`인물 1 (왼쪽)` 등)
- `compute_square_face_box`(face_utils): 정사각형 크롭, 이미지 경계 내 클램프. 구도 두 가지:
  - `FACE_CROP_SCALE/EYE`(얼굴 ×1.3, 눈 위 42%): 자르기 화면·추가 사진 자동 크롭용(`padded_box`). 1.65배는 너무 넓다는 피드백으로 변경. 1.2배까지 줄여도 얼굴 재검출·ArcFace 특징이 유지됨을 확인.
  - `HEAD_CROP_SCALE/EYE`(×1.65, 눈 위 38%): 연예인 대표 사진과 CLIP 시각 비교용. 바꾸면 대표 사진과 `celebrity_visual.npz`를 다시 만들어야 합니다.

---

## 5. 프론트엔드 규칙 (`static/`)

- **Base URL**: `getBaseUrl()`이 경로가 `/facematching`으로 시작하면 prefix를 붙입니다. 모든 `fetch`는 `BASE_URL + '/api/...'` 형태로 호출하세요.
- `index.html`은 `style.css`/`app.js`를 **`?v=Date.now()` 캐시 버스팅**으로 동적 로드합니다. 파일명을 바꾸면 이 로더도 수정해야 합니다.
- 상태는 `app.js` 상단 전역 변수로 관리합니다. 사진 대상 식별자는 `1`, `2`, `'celeb'`이며 크롭/웹캠/얼굴 선택 함수는 이 `targetId`를 받습니다.
- UI 이벤트는 HTML 인라인 `onclick="fn()"`으로 연결되어 있으므로, 함수명 변경 시 `index.html`도 함께 수정하세요.
- 서버 데이터를 `innerHTML`에 넣을 때는 반드시 `escapeHtml()`을 거칩니다(연예인 이름은 Wikidata 라벨).
- 아이콘은 Lucide — DOM을 동적으로 추가한 뒤에는 `lucide.createIcons()` 호출.
- 카메라 촬영은 휴대폰 카메라 앱과 같게 동작합니다: 전면 카메라는 미리보기를 거울 모드(`scaleX(-1)`)로 보여 주고 **보이는 그대로** 저장, 후면 카메라는 반전 없음. 카메라가 2대 이상이면 전면/후면 전환 버튼이 보입니다(`startWebcamStream`, `switchWebcamFacing`).
- 현재 UI는 비교 모드를 항상 `family`로 보냅니다. `celebrity`/`identical` 모드는 백엔드에만 있습니다.
- **디자인 시스템 (`style.css` 하단)**: 모바일 우선, 짙은 남색 배경 + 인디고/바이올렛 한 가지 강조색(핑크는 연예인 점수에만). 새 UI는 아래 공통 클래스를 쓰고, 긴 Tailwind 클래스 문자열을 JS에서 직접 넣지 않습니다.
  - `.card`(16px 라운드 카드), `.btn` + `.btn-primary/-secondary/-ghost`, 크기 `.btn-lg/-sm/-icon` — 버튼 글자는 줄바꿈되지 않습니다(`white-space: nowrap`).
  - `.seg` + `.seg-btn`(탭·성별 필터·자르기 비율): 활성 상태는 JS에서 `classList.toggle('is-active', …)`로만 바꿉니다.
  - `.chip`(+`-amber`, `-pink`), `.part-label/.part-track/.part-fill`(부위별 막대), `details.accordion`(접는 섹션), `.no-scrollbar`.
  - 본문은 `word-break: keep-all`(한국어를 단어 단위로 줄바꿈).
- 헤더는 모든 화면 크기에서 한 줄입니다. 서버 상태는 휴대폰에서 점만, PC에서 짧은 문구(`setServerStatus`)로 표시합니다.
- **결과 이미지 카드** (`app.js` 하단 "Result image card"): 결과만 담은 1080px 폭 세로 PNG를 캔버스로 직접 그립니다(외부 라이브러리 없음). `saveResultImage(kind)`는 다운로드, `shareResultImage(kind)`는 `navigator.share({files})`(미지원 기기는 저장으로 대체). iOS Safari는 탭 직후에만 공유 창을 허용하므로 결과가 나오면 `prepareResultCard(kind)`로 **미리 렌더링**해 둡니다. 연예인 사진은 같은 출처(`/facematching/static/...`)라 캔버스가 오염되지 않습니다 — 외부 URL 이미지를 넣지 마세요.
- **닮음 강도** (`celebrity_match_strength`, 응답 `match_strength`): 1위 코사인을 "다른 사람들의 1위 닮은꼴" 분포(`CELEB_TOP1_PERCENTILES`)의 백분위로 바꿔 라벨·설명을 만듭니다. 연예인 % 환산(`celebrity` 앵커)도 같은 분포에 맞춰 두어 점수와 라벨이 어긋나지 않습니다. DB를 크게 바꾸면 두 표를 함께 다시 측정하세요.
- 메인 페이지(`/`, `/facematching/`)는 `Cache-Control: no-cache`로 내려갑니다. 배포 직후 예전 HTML이 보이는 문제 때문입니다.

---

## 6. 연예인 DB 관리

```bash
python populate_celebrities.py --workers 4   # 약 2천 명, 캐시 없으면 1.5~2시간 (진행 로그는 표준출력)
```
- 후보 = `CELEBRITY_CATALOG` 시드(항상 포함, 손으로 쓴 `face_type`/`vibe` 설명 재사용) + Wikidata의 생존 한국 연예인(배우·가수·래퍼·뮤지션·방송인·유튜버·성우·댄서·모델 등) 중 Commons 카테고리 **또는** 대표 이미지(P18)가 있는 사람(약 3,500명).
- 출생 연대·성별별 할당량(`DECADE_QUOTA`)을 둘 수 있습니다(현재는 1940~2000년대 전원). 인지도순 상한만 두면 20대 아이돌 위주가 되어 30대 이상 사용자에게 닮은 얼굴이 없었습니다.
- 사람마다 최대 15장(`PHOTOS_PER_PERSON`, 1280px)을 받아 `data/celeb_cache/<QID>/`에 캐시합니다. 출처(`source`): `main`(Wikidata 대표 이미지), `category`(Commons 카테고리 트리 `deepcat`), `search`(파일 제목·설명에 이름이 들어간 사진 — 한글 3글자 이상 이름, 공백/하이픈 있는 영문 이름만).
- **재실행 시 캐시를 재사용**합니다. 이전 빌드에서 일치 사진이 `TOPUP_BELOW_MATCHED`(6)장 미만이던 사람만 부족분을 추가로 검색·다운로드합니다.
- **사진 수의 한계**: 덜 알려진 인물은 Commons에 자유 라이선스 사진 자체가 거의 없습니다. 이름 검색 결과 대부분은 동명이인·무관한 행사 사진이라 얼굴 대조에서 걸러집니다. 뉴스 사진 등 저작권 있는 출처는 쓰지 않습니다.
- 본인 식별은 **여러 사진에 반복 등장하는 얼굴**(ArcFace 코사인 ≥ 0.40)로 합니다. 기준 얼굴(anchor)은 `main`/`category` 사진에서만 고르고, `search` 사진은 그 얼굴과 일치할 때만 평균에 들어갑니다(동명이인 방지). 얼굴 폭 36px 이상이면 사용합니다(그룹 사진 속 본인 얼굴이 대부분 40~60px라서 60px 기준에서는 버려졌음).
- 대표 사진은 일치 얼굴 중 크고 전형적인 것을 골라 얼굴 중심으로 크롭해 `static/celebrities/`에 씁니다. 원본 출처는 `photo_source`(Commons 파일 페이지)에 남습니다.
- **Wikimedia 요청 제한**: 모든 요청은 공유 속도 제한(`MIN_REQUEST_INTERVAL` 0.35초)을 거치고 429의 `Retry-After`를 따릅니다. 병렬 작업을 여러 개 띄우면 429가 나므로 수집 스크립트는 한 번에 하나만 돌리세요.
- 시드 추가: `CELEBRITY_CATALOG`에 `name, query, file, gender, category, face_type, vibe`(선택: `lang`, `en_query`). `query`는 한국어 위키백과 검색어로 Wikidata 항목을 찾는 데 씁니다.
- DB 교체 후에는 서비스 재시작이 필요합니다(`load_celebrity_db` 모듈 캐시).

---

## 7. 코딩 컨벤션

- **사용자 노출 문자열(에러 메시지, 판정 문구, 프롬프트)은 한국어**, 코드·로그·docstring은 영어를 기본으로 합니다.
- 로깅은 `logging.getLogger("facematch")` / `"facematch.celebrity"`를 사용합니다. `print`는 오프라인 스크립트에서만.
- 여러 모듈에서 쓰는 로직(이미지 정규화, 얼굴 검출·임베딩, 점수 환산, JSON 추출, 모델 경로)은 **`face_utils.py`에만** 두고 import 해서 쓰세요.
  - `app.py`의 `process_image_to_base64` / `extract_json_from_response`는 공용 함수를 감싸 HTTP 400 변환과 30점 폴백만 추가하는 얇은 래퍼입니다.
- LLM 응답은 절대 신뢰하지 말고 항상 파싱 폴백 + 숫자 변환 + 범위 클램프를 거치세요. LLM이 실패해도 기능이 동작하도록 기본 문구를 둡니다.
- LLM 프롬프트는 JSON 스키마를 문자열로 명시하고 "JSON으로만 출력"을 강제하는 기존 스타일을 유지합니다.
- `celebrity_service`는 `app.py`에서 **엔드포인트 내부에서 지연 import** 합니다. 이 구조를 유지하세요.
- 커밋 메시지: `feat:` / `fix:` 접두사 + 한국어 또는 영어 요약 (기존 히스토리 스타일).

---

## 8. 배포 (112 서버)

- 호스트: `192.168.219.112` (SSH 별칭 `local-ai-server`), GPU RTX 2070 SUPER (Gemma 전용, ArcFace는 CPU)
- 배포 경로: `/home/lmo0317/apps/facematch` (venv: `venv/`, git 저장소 아님)
- 서비스: `facematch.service` (systemd **user** service), `llama-gemma4.service`에 의존
- `llama-gemma4.service`(`~/.config/systemd/user/`)는 `--no-mmproj-offload` 없이 실행해 이미지 인코더(mmproj)를 GPU에서 돌립니다(이미지당 prefill 3.6초 → 1.3초, VRAM 약 4.8GB 사용). 되돌리려면 같은 폴더의 `llama-gemma4.service.bak-before-mmproj-offload`를 복원하고 `daemon-reload` 후 재시작하세요.
- 외부 접근: 리버스 프록시 `https://minohlee.mooo.com/facematching/` → `:8501`
- 별도 배포 스크립트는 없습니다. 파일을 복사(scp)한 뒤 재시작합니다. `data/`(빌드 캐시)는 올리지 않습니다.
  ```bash
  ssh local-ai-server "systemctl --user restart facematch.service"
  ssh local-ai-server "journalctl --user -u facematch.service -f"
  ```
- 서버에 `models/arcface_w600k_r50.onnx`와 venv의 `onnxruntime`이 있어야 합니다. 없으면 두 기능 모두 얼굴 인식 점수를 낼 수 없습니다.
- `models/clip_vit_b32_vision_q.onnx`(HuggingFace `Xenova/clip-vit-base-patch32`의 `onnx/vision_model_quantized.onnx`)가 없으면 시각 재정렬 없이 ArcFace 순서로만 동작합니다.
- 대표 사진을 바꾸거나 DB를 다시 만들면 `python populate_celebrities.py --visual-only`로 `celebrity_visual.npz`도 갱신하세요(전체 빌드 시에는 자동).

---

## 9. 알려진 이슈 / 주의사항

- **닮은 연예인 검색의 한계(실측)**: 사람들이 닮았다고 말하는 쌍 68개 기준, 사진 1장일 때 상대가 TOP 3에 드는 비율 약 18%, TOP 10 약 29%(무작위면 중앙 순위 약 150위 → 현재 27위). 얼굴형 비율(106 랜드마크)·나이·CLIP(전체 인상)·glintr100 모델을 섞어 봤지만 개선이 없었습니다. 효과가 확인된 것은 **사진 여러 장 평균**(중앙 순위 36→18위)과 **DB 다양화**입니다.

- OpenCV는 서버와 동일하게 `opencv-python-headless`를 사용합니다. 같은 venv에 `opencv-python`을 함께 설치하면 `cv2`가 충돌합니다.
- InsightFace 사전학습 모델(`w600k_r50`)은 **비상업·연구용 라이선스**입니다. 상업 서비스로 전환 시 교체를 검토하세요.
- 연예인 사진은 Wikimedia Commons(CC 라이선스) 출처입니다. 출처 URL은 DB의 `photo_source`에 있습니다.
- ArcFace는 "같은 사람인가"를 학습한 모델이라, 성별·나이 차가 큰 가족(엄마-아들 등)은 실제 닮음보다 코사인이 낮게 나오는 경향이 있습니다.
- CORS가 `allow_origins=["*"]`로 전체 개방되어 있습니다.
- `/api/samples`의 이미지 경로는 prefix 없는 `/static/...`이며, 프론트에서 `BASE_URL`을 붙여 사용합니다.
