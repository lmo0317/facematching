# CLAUDE.md — FaceMatch AI 프로젝트 지침

이 문서는 이 저장소에서 작업할 때 따라야 할 구조·규칙·주의사항을 정리한 개발 지침입니다.
사용자용 소개/접속 정보는 [README.md](README.md)를 참고하세요.

---

## 1. 프로젝트 개요

Gemma 4 E4B 멀티모달(Vision) LLM + OpenCV 얼굴 인식 모델을 이용한 **얼굴 닮음 분석 웹앱**입니다.
핵심 기능은 두 가지입니다.

| 탭 | 기능 | API |
|---|---|---|
| `compare` | **사진 2장 닮음 분석** (가족·붕어빵 지수) | `POST /api/compare-json` |
| `celeb` | **내 사진으로 닮은 연예인 찾기** (TOP 3 + 실사진) | `POST /api/find-celebrity` |

- 백엔드: Python 3.12, FastAPI + Uvicorn (단일 프로세스, 빌드 단계 없음)
- 프론트엔드: 순수 HTML/JS (프레임워크·번들러 없음), Tailwind/Lucide/Cropper.js는 CDN 로드
- LLM: 로컬 `llama-server`(llama.cpp)의 OpenAI 호환 `/v1/chat/completions` 엔드포인트
- 얼굴 검출/임베딩: OpenCV YuNet(검출) + SFace(128-D 임베딩)

---

## 2. 디렉터리 구조

```
app.py                     # FastAPI 앱: 라우트, 얼굴 검출, 2장 비교 로직, 점수 보정
celebrity_service.py       # 닮은 연예인 탐색 엔진 (app.py에서 지연 import)
face_utils.py              # 공용 헬퍼: 이미지→data URL, LLM JSON 추출, SFace 임베딩, 위키백과 사진 다운로드
populate_celebrities.py    # 오프라인 스크립트: 연예인 사진 다운로드 + SFace 임베딩 DB 생성
celebrity_db.json          # 연예인 메타데이터 + 128-D 임베딩 (populate 스크립트 산출물, ~82명)
models/
  face_detection_yunet.onnx    # 얼굴 검출 (git 추적됨 — 강제 add)
  face_recognition_sface.onnx  # 얼굴 임베딩 (gitignore 대상, 서버에 수동 배치 필요)
static/
  index.html               # 단일 페이지 UI (탭 2개 + 크롭/웹캠 모달)
  js/app.js                # 전역 상태 + 모든 UI 로직 (~1250줄)
  css/style.css            # Tailwind 보조 커스텀 스타일
  samples/                 # 비교 탭 예제 프리셋 이미지
  celebrities/             # 연예인 사진 캐시 (600x600 JPEG)
facematch.service          # systemd user service 정의
requirements.txt
```

---

## 3. 실행 & 환경 변수

```bash
pip install -r requirements.txt
python app.py            # 또는: uvicorn app:app --host 0.0.0.0 --port 8501
```

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8081` | Gemma 4 llama-server 주소 |
| `MODEL_NAME` | `gemma-4-e4b-it-q4km` | 요청 payload의 model 이름 |
| `HOST` / `PORT` | `0.0.0.0` / `8501` | `python app.py` 실행 시에만 사용 |

- 로컬(Windows)에는 LLM 서버가 없으므로 비교/연예인 기능은 112 서버에서만 실제 동작합니다. 로컬에선 `/api/health`가 `unreachable`로 나오는 것이 정상입니다.
- LLM을 로컬에서 쓰려면 SSH 터널로 112 서버의 8081을 포워딩한 뒤 `LLAMA_SERVER_URL`을 지정하세요.
- 자동화 테스트는 없습니다. 변경 검증은 `/api/health`, 예제 프리셋(`/api/samples`)을 이용한 수동 확인으로 합니다.

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
| POST | `/api/find-celebrity` | 닮은 연예인 TOP 3 |

### 4.2 사진 2장 비교 파이프라인 (`execute_face_comparison`)
1. 모드(`family` 기본 / `celebrity` / `identical`)별 한국어 프롬프트 구성 → 두 이미지와 함께 Gemma 4에 전송 (`temperature 0.05`, timeout 90s)
2. `extract_json_from_response`로 응답에서 JSON 추출 (think 태그 제거 → 코드블록 → 최외곽 중괄호 → 실패 시 30점 고정 폴백)
3. **점수 보정**:
   - 부위별 5개 점수(`eyes, nose, mouth, face_shape, features`) 각각 `calibrate_similarity_score` 적용
   - 종합 raw = `0.4 × LLM 종합점수 + 0.6 × 가중평균(눈 .25, 코 .20, 입 .20, 얼굴형 .20, 특징 .15)` → 보정
   - `get_verdict_for_score(score, mode)`로 판정 문구 부여
4. 에러 매핑: LLM 비정상 응답 502 / 타임아웃 504 / 연결 실패 503 / 기타 500

> `calibrate_similarity_score`의 구간표(≤40→10~25, 40~50→25~45, 50~65→45~76, 65~85→76~92, >85→92~99)는 "남남 과대채점 방지"를 위해 튜닝된 값입니다. 조정 시 예제 3(남남, 35% 이하)과 예제 1·2(가족, 85% 이상) 기대치를 함께 확인하세요.

### 4.3 닮은 연예인 파이프라인 (`celebrity_service.execute_celebrity_lookalike`)
1. 이미지를 768px로 정규화 → Gemma 4에 "안면 형태학 분석 + 한국 연예인 TOP 추천" JSON 요청 (`temperature 0.25`, timeout 45s, `cache_prompt: False`)
2. **Gemma 실패 시 폴백**: 사용자 SFace 임베딩과 `celebrity_db.json` 임베딩의 코사인 유사도로 순위 산정
3. 상위 3명에 대해 `resolve_celebrity_image`로 사진 확보:
   DB 매칭 → 로컬 파일(`<정규화된 이름>.jpg`) → **위키백과(ko→en) 실시간 검색·다운로드·600px 정사각 크롭 후 캐시**
4. 사진이 있으면 SFace 코사인 유사도로 점수 보정: `bonus = round((sim - 0.25) × 25)`, 최종 62~96으로 클램프
5. 점수 내림차순 재정렬 후 `top_celebrity`, `candidates`, `all_celebrities`, `face_features` 반환
- 사진을 못 찾으면 `gong_yoo.jpg`가 대체 이미지로 사용됩니다.

### 4.4 얼굴 검출 (`detect_faces_in_image`)
- EXIF 회전 보정 후 YuNet(score 0.55) → 실패 시 Haar Cascade 폴백
- 얼굴은 x좌표(왼→오) 순 정렬, 라벨은 `인물 1 (왼쪽)` 식으로 생성
- `compute_square_face_box`: 얼굴 크기 ×1.65 정사각형, 눈 위치가 위에서 38% 지점에 오도록 배치, 이미지 경계 내 클램프

---

## 5. 프론트엔드 규칙 (`static/`)

- **Base URL**: `getBaseUrl()`이 경로가 `/facematching`으로 시작하면 prefix를 붙입니다. 모든 `fetch`는 `BASE_URL + '/api/...'` 형태로 호출하세요. 절대경로 하드코딩 금지.
- `index.html`은 `style.css`/`app.js`를 **`?v=Date.now()` 캐시 버스팅**으로 동적 로드합니다. 파일명을 바꾸면 이 로더도 수정해야 합니다.
- 상태는 `app.js` 상단의 전역 변수(`photo1Data`, `photoCelebData`, `currentMode`, `detectedFaces1` 등)로 관리합니다. 사진 대상 식별자는 `1`, `2`, `'celeb'` 세 가지이며, 크롭/웹캠/얼굴 선택 함수는 모두 이 `targetId`를 인자로 받습니다.
- UI 이벤트는 HTML의 인라인 `onclick="fn()"`으로 연결되어 있으므로, 함수명 변경 시 `index.html`도 함께 수정하세요.
- 아이콘은 Lucide — DOM을 동적으로 추가한 뒤에는 `lucide.createIcons()`를 호출해야 렌더링됩니다.
- 이미지 입력: 드래그앤드롭, 파일 선택, `Ctrl+V` 붙여넣기, 웹캠. 업로드 전 `compressImage`(최대 800px)로 압축, 얼굴 자동 검출 후 Cropper.js 1:1 크롭 지원.
- 현재 UI는 비교 모드를 항상 `family`로 보냅니다. `celebrity`/`identical` 모드는 백엔드에만 남아 있습니다.
- 스타일은 Tailwind 유틸리티 클래스 중심의 다크 테마(slate/indigo/pink 계열, `glass-card`)를 따릅니다.

---

## 6. 연예인 DB 관리

- 연예인 추가/수정: `populate_celebrities.py`의 `CELEBRITY_CATALOG`에 항목(`name, query, file, gender, category, face_type, vibe`, 선택적으로 `lang: "en"` 또는 영문 폴백 검색어 `en_query`)을 추가 후 실행
  ```bash
  python populate_celebrities.py
  ```
  → 누락된 사진만 위키백과에서 다운로드 → 전체 DB(`celebrity_db.json`)를 **재생성(덮어쓰기)** 합니다.
- `file`은 영문 snake_case(`kim_soo_hyun.jpg`)를 사용합니다. 위키백과 검색어가 모호하면 `query`에 `"(배우)"`, `"(가수)"` 등 동음이의 구분자를 붙입니다.
- 4KB 이하 파일은 유효하지 않은 이미지로 간주되어 재다운로드 대상입니다.
- 런타임 on-demand 다운로드는 **한글 파일명**(`김민재.jpg` 등)으로 `static/celebrities/`에 저장됩니다. 이 파일들은 git에 untracked로 쌓이므로, 커밋 여부를 의도적으로 결정하세요(DB에 정식 편입하려면 catalog에 영문 파일명으로 추가).
- `celebrity_db.json`은 모듈 레벨에서 캐시(`_CACHED_DB`)되므로 DB 갱신 후에는 서비스 재시작이 필요합니다.

---

## 7. 코딩 컨벤션

- **사용자 노출 문자열(에러 메시지, 판정 문구, 프롬프트)은 한국어**, 코드·로그·docstring은 영어를 기본으로 합니다.
- 로깅은 `logging.getLogger("facematch")` / `"facematch.celebrity"`를 사용합니다. `print`는 오프라인 스크립트에서만.
- 이미지 처리 공통 규칙: `ImageOps.exif_transpose` → RGB 변환 → 긴 변 768px 리사이즈 → JPEG q88 → data URL.
- 여러 모듈에서 쓰는 로직(이미지 정규화, JSON 추출, SFace 임베딩, 위키백과 다운로드, 모델 경로)은 **`face_utils.py`에만** 두고 import 해서 쓰세요. 복사해서 중복 구현하지 않습니다.
  - `app.py`의 `process_image_to_base64` / `extract_json_from_response`는 공용 함수를 감싸 HTTP 400 변환과 30점 폴백만 추가하는 얇은 래퍼입니다.
- LLM 응답은 절대 신뢰하지 말고 항상 파싱 폴백 + 점수 `int(round(float(x)))` 변환 + 범위 클램프를 거치세요.
- LLM 프롬프트는 JSON 스키마를 문자열로 명시하고 "JSON으로만 출력"을 강제하는 기존 스타일을 유지합니다.
- `celebrity_service`는 `app.py`에서 **엔드포인트 내부에서 지연 import** 합니다(순환 의존 없음, 모듈 로딩 비용 분리 목적). 이 구조를 유지하세요.
- 커밋 메시지: `feat:` / `fix:` 접두사 + 한국어 또는 영어 요약 (기존 히스토리 스타일).

---

## 8. 배포 (112 서버)

- 호스트: `192.168.219.112` (SSH 별칭 `local-ai-server`), GPU RTX 2070 SUPER
- 배포 경로: `/home/lmo0317/apps/facematch` (venv: `venv/`)
- 서비스: `facematch.service` (systemd **user** service), `llama-gemma4.service`에 의존
- 외부 접근: 리버스 프록시 `https://minohlee.mooo.com/facematching/` → `:8501`
- 별도 배포 스크립트는 없습니다. 파일 반영 후 재시작:
  ```bash
  ssh local-ai-server "systemctl --user restart facematch.service"
  ssh local-ai-server "journalctl --user -u facematch.service -f"
  ```
- 서버에 `models/face_recognition_sface.onnx`가 있어야 SFace 보정/폴백이 동작합니다(없으면 조용히 비활성화됨).

---

## 9. 알려진 이슈 / 주의사항

- OpenCV는 서버와 동일하게 `opencv-python-headless`를 사용합니다. 같은 venv에 `opencv-python`을 함께 설치하면 `cv2`가 충돌하니 둘 중 하나만 설치하세요.
- `.gitignore`는 `models/*.onnx`를 제외하되 YuNet만 예외로 추적합니다. SFace 모델은 저장소에 없습니다.
- `CELEBRITY_CATALOG`는 `file` 기준으로 중복 없이 유지하세요(DB 빌드 시 중복은 경고 후 건너뜀). 한국어 위키 검색이 실패하는 인물은 별도 항목 대신 `en_query` 필드로 영문 폴백 검색어를 지정합니다.
- CORS가 `allow_origins=["*"]`로 전체 개방되어 있습니다.
- `/api/samples`의 이미지 경로는 prefix 없는 `/static/...`이며, 프론트에서 `BASE_URL`을 붙여 사용합니다.
