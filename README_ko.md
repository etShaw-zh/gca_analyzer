# GCA Analyzer (Need to be determined by native speakers)

고급 NLP 기술과 정량적 지표를 사용하여 그룹 대화 동역학을 분석하는 Python 패키지입니다.

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | 한국어

## 특징

- **다국어 지원**: 고급 LLM 모델을 통한 한국어를 포함한 다국어 지원
- **포괄적인 지표**: 다차원적 그룹 상호작용 분석
- **자동 분석**: 최적의 분석 윈도우를 찾고 상세한 통계 생성
- **유연한 구성**: 다언한 분석 요구에 맞춤 설정 가능한 매개변수
- **쉬운 통합**: 명령줄 인터페이스와 Python API 지원

## 빠른 시작

### 설치

```bash
# PyPI에서 설치
pip install gca_analyzer

# 개발용
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 기본 사용법

1. 대화 데이터를 CSV 형식으로 준비 (필수 열 포함):
```
conversation_id,person_id,time,text
1A,student1,0:08,선생님 안녕하세요!
1A,teacher,0:10,여러분 안녕하세요!
```

2. 분석 실행:
```bash
python -m gca_analyzer --data your_data.csv
```

3. GCA 측정의 기술 통계:

![기술 통계](/doc/imgs/gca_results.jpg)

## 상세 사용법

### 명령줄 옵션

```bash
python -m gca_analyzer --data <path_to_data.csv> [options]
```

#### 필수 인자
- `--data`: 대화 데이터 CSV 파일 경로

#### 선택적 인자
- `--output`: 결과 출력 디렉토리 (기본값: `gca_results`)
- `--best-window-indices`: 윈도우 크기 최적화 임계값 (기본값: 0.3)
  - 범위: 0.0-1.0
  - 낮은 값은 더 작은 윈도우 크기로 이어짐
- `--console-level`: 로깅 레벨 (기본값: INFO)
  - 옵션: DEBUG, INFO, WARNING, ERROR, CRITICAL
- `--model-name`: 텍스트 처리를 위한 LLM 모델
  - 기본값: `iic/nlp_gte_sentence-embedding_chinese-base`
- `--model-mirror`: 모델 다운로드 미러
  - 기본값: `https://modelscope.cn/models`

### 입력 데이터 형식

필수 CSV 열:
- `conversation_id`: 각 대화의 고유 식별자
- `person_id`: 참가자 식별자
- `time`: 메시지 날짜 (형식: YYYY-MM-DD HH:MM:SS 또는 HH:MM:SS 또는 MM:SS)
- `text`: 메시지 내용

### 출력 지표

분석기는 다음 측정 항목에 대한 포괄적인 통계를 생성합니다:

1. **참여도**
   - 상대적 기여 빈도 측정
   - 음수 값은 평균 이하 참여를 나타냄
   - 양수 값은 평균 이상 참여를 나타냄

2. **응답성**
   - 참가자들이 다른 사람에게 응답하는 정도를 측정
   - 높은 값은 더 나은 응답 행동을 나타냄

3. **내부 응집성**
   - 개인 기여의 일관성을 측정
   - 높은 값은 더 일관된 메시징을 나타냄

4. **사회적 영향력**
   - 그룹 토론에 대한 영향력을 측정
   - 높은 값은 다른 사람들에 대한 강한 영향력을 나타냄

5. **새로움**
   - 새로운 내용의 도입을 측정
   - 높은 값은 더 혁신적인 기여를 나타냄

6. **커뮤니케이션 밀도**
   - 메시지당 정보 내용을 측정
   - 높은 값은 정보가 더 풍부한 메시지를 나타냄

결과는 지정된 출력 디렉토리에 CSV 파일로 저장됩니다.

## 자주 묻는 질문

1. **Q: 왜 일부 참여도 값이 음수인가요?**
   A: 참여도 값은 그룹 크기를 기반으로 조정되며, 완전히 동등한 참여로부터의 편차를 나타냅니다. 음수 값은 동등 참여량 이하의 기여를, 양수 값은 동등 참여량 이상의 기여를 나타냅니다. 값이 0인 경우는 모든 참가자가 동등하게 기여했음을 의미합니다. 이러한 측정 방식을 통해 각 참가자의 동등 참여 대비 상대적 성과를 직관적으로 파악할 수 있습니다.

2. **Q: 최적의 윈도우 크기는 무엇인가요?**
   A: 분석기는 `best-window-indices` 매개변수를 기반으로 최적의 윈도우 크기를 자동으로 찾습니다. 낮은 값(예: 0.03)은 더 작은 윈도우로 이어지며, 산발적인 대화에 더 적합할 수 있습니다.

3. **Q: 다른 언어는 어떻게 처리되나요?**
   A: 분석기는 텍스트 처리를 위해 LLM 모델을 사용하며 기본적으로 여러 언어를 지원합니다. 중국어 텍스트의 경우 중국어 기본 모델을 사용합니다.

## 기여하기

GCA Analyzer에 기여해 주셔서 감사합니다! 다음과 같은 방법으로 참여하실 수 있습니다:

### 기여 방법
- [GitHub Issues](https://github.com/etShaw-zh/gca_analyzer/issues)를 통해 버그 신고 및 기능 요청
- 버그 수정 및 기능 추가를 위한 Pull Request 제출
- 문서 개선
- 사용 사례 및 피드백 공유

### 개발 환경 설정
1. 저장소 Fork
2. Fork 클론:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gca_analyzer.git
   cd gca_analyzer
   ```
3. 개발 의존성 설치:
   ```bash
   pip install -e ".[dev]"
   ```
4. 변경사항을 위한 브랜치 생성:
   ```bash
   git checkout -b feature-or-fix-name
   ```
5. 변경사항 커밋:
   ```bash
   git add .
   git commit -m "변경사항 설명"
   ```
6. Push하고 Pull Request 생성

### Pull Request 가이드라인
- 기존 코드 스타일 준수
- 새로운 기능에 대한 테스트 추가
- 필요한 문서 업데이트
- 모든 테스트 통과 확인
- Pull Request는 하나의 변경사항에 집중

## 라이선스

Apache 2.0

## 인용

연구에서 이 도구를 사용할 경우 다음을 인용해 주세요:

```bibtex
@software{gca_analyzer2025,
  author = {Xiao, Jianjun},
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
