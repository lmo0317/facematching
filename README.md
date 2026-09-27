# 🎭 FaceMatch AI - Gemma 4 E4B 안면 유사도 정밀 대조 시스템

112 서버(192.168.219.112)에 탑재된 **Google Gemma 4 E4B 멀티모달(Vision) LLM**을 활용하여 사진 2장을 정밀 대조하고 유사도 점수와 이목구비 골격 감정 소견서를 생성하는 웹 애플리케이션입니다.

---

## 🌟 주요 기능

1. **Gemma 4 E4B Vision 기반 정밀 안면 대조**:
   - 112 서버의 NVIDIA GeForce RTX 2070 SUPER GPU 가속 및 MTMD(Multimodal) 프로젝터 연동
   - 초당 100+ 토큰의 고속 추론으로 얼굴형, 눈매, 콧대, 입술, 고유 생체 특징 정밀 대조
2. **다양한 사진 입력 지원**:
   - 파일 드래그 앤 드롭 (Drag & Drop) 및 파일 탐색기 선택
   - 클립보드 이미지 직접 붙여넣기 (`Ctrl + V`) 지원
   - 웹캠 실시간 카메라 촬영 지원
3. **1-클릭 빠른 테스트 예제 프리셋**:
   - **예제 1**: 동일 인물 대조 (스튜디오 정면 vs 야외 미소 사진) - 85% 이상 동일 인물 유력 판정
   - **예제 2**: 다른 인물 대조 (남성 A vs 남성 B) - 40% 이하 다른 인물 판정
   - **예제 3**: 성별/안경 대조 (여성 A vs 남성 B) - 성별 및 골격 차이 정확 식별
4. **시각화된 정밀 분석 리포트**:
   - 0~100% 원형 게이지 유사도 스코어 (색상 그라데이션)
   - 5대 핵심 부위별 일치도 바 차트 (얼굴형, 눈매, 콧대, 입술, 고유특징)
   - 주요 공통점 (Similarities) 및 차이점 (Differences) 체크리스트
   - 촬영 환경/외적 변수(조명, 각도, 표정, 안경 등) 영향 분석
   - 전문 국과수 수준 종합 감정 소견서
   - 감정서 원클릭 클립보드 복사 기능

---

## 🚀 배포 정보 (112 서버)

- **접속 주소**: [http://192.168.219.112:8501](http://192.168.219.112:8501)
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
