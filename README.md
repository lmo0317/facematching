# 🎭 FaceMatch AI - 얼굴 닮음 분석 (ArcFace + Gemma 4 E4B)

112 서버(192.168.219.112)에서 동작하는 얼굴 닮음 분석 웹 애플리케이션입니다.
**얼굴 인식 모델(InsightFace ArcFace)이 점수와 닮은 연예인 후보를 정하고**, **Google Gemma 4 E4B 멀티모달 LLM이 닮은 부위와 소견을 설명**합니다.

---

## 🌟 주요 기능

1. **사진 2장 닮음 분석 (가족·붕어빵 지수)**:
   - 두 얼굴의 ArcFace 512차원 특징 유사도를 실제 가족/남남 사진 분포로 보정해 % 점수 산출 (80%)
   - Gemma 4의 부위별 시각 판단 반영 (20%) 및 부위별 점수·공통점·차이점·종합 소견 작성
   - LLM 서버가 응답하지 않아도 얼굴 인식 점수는 표시
2. **내 사진으로 닮은 연예인 찾기**:
   - Wikidata·Wikimedia Commons에서 수집한 한국 연예인 수백 명, 1인당 여러 장의 사진으로 만든 얼굴 특징 DB에서 가장 가까운 TOP 3 선정
   - 성별 필터(자동/남성/여성), Gemma 4가 사진을 비교해 닮은 포인트 설명
3. **다양한 사진 입력 지원**: 드래그 앤 드롭, 파일 선택, 클립보드 붙여넣기(`Ctrl + V`), 웹캠 촬영, 단체 사진 얼굴 선택·크롭
4. **시각화 리포트**: 원형 게이지 점수, 5대 부위별 바 차트, 공통점/차이점, 종합 소견, 클립보드 복사

---

## 🚀 배포 및 접속 정보 (112 서버)

- **메인 포털(Service Hub)**: [http://192.168.219.112](http://192.168.219.112) 또는 [https://minohlee.mooo.com](https://minohlee.mooo.com)
- **전용 서비스 경로**: [http://192.168.219.112/facematching/](http://192.168.219.112/facematching/) 또는 [https://minohlee.mooo.com/facematching/](https://minohlee.mooo.com/facematching/)
- **포트 직접 접속**: [http://192.168.219.112:8501](http://192.168.219.112:8501)
- **배포 위치**: `/home/lmo0317/apps/facematch`
- **Gemma 4 LLM 엔드포인트**: `http://127.0.0.1:8081` (`llama-gemma4.service`)
  - 모델: `/home/lmo0317/models/gemma-4-E4B-it-Q4_K_M.gguf`
  - 멀티모달 프로젝터: `/home/lmo0317/models/mmproj-gemma-4-E4B-F16.gguf`
- **시스템 서비스**: `facematch.service` (systemd user service)

---

## 🛠️ 관리 명령어

```bash
# 서비스 상태 확인
ssh local-ai-server "systemctl --user status facematch.service"

# 서비스 재시작
ssh local-ai-server "systemctl --user restart facematch.service"

# 실시간 로그 확인
ssh local-ai-server "journalctl --user -u facematch.service -f"

# Gemma 4 LLM 서버 상태 확인
ssh local-ai-server "systemctl --user status llama-gemma4.service"
```
