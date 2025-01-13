# GCA Analyzer

NLP 기술과 정량적 지표를 사용하여 그룹 대화를 분석하는 Python 패키지입니다.

[English](README.md) | [中文](README_zh.md) | [日本語](README_ja.md) | 한국어

## 특징

- **다국어 지원**: LLM 모델을 통한 중국어 및 기타 언어 내장 지원
- **포괄적인 지표**: 다차원적 그룹 상호작용 분석
- **자동화된 분석**: 최적의 분석 윈도우를 자동으로 찾고 상세한 통계 생성
- **유연한 설정**: 다양한 분석 요구에 맞춤 가능한 매개변수
- **쉬운 통합**: 명령줄 인터페이스와 Python API 지원

## 빠른 시작

### 설치

```bash
# PyPI에서 설치
pip install gca_analyzer

# 개발용 설치
git clone https://github.com/etShaw-zh/gca_analyzer.git
cd gca_analyzer
pip install -e .
```

### 기본 사용법

1. 필수 열이 포함된 CSV 형식의 대화 데이터 준비:
```
conversation_id,person_id,time,text
1A,student1,0:08,선생님 안녕하세요!
1A,teacher,0:10,여러분 안녕하세요!
```

2. 분석 실행:
```bash
python -m gca_analyzer --data your_data.csv
```

3. GCA 지표의 기술 통계:

분석기는 다음 지표에 대한 포괄적인 통계를 생성합니다:

![기술 통계](/docs/_static/gca_results.jpg)

- **참여도**
   - 상대적 기여 빈도를 측정
   - 음수 값은 평균 이하의 참여를 나타냄
   - 양수 값은 평균 이상의 참여를 나타냄

- **응답성**
   - 참가자들이 다른 사람에게 응답하는 정도를 측정
   - 높은 값은 더 나은 응답 행동을 나타냄

- **내부 응집성**
   - 개인 기여의 일관성을 측정
   - 높은 값은 더 일관된 메시징을 나타냄

- **사회적 영향력**
   - 그룹 토론에 대한 영향력을 측정
   - 높은 값은 다른 사람들에 대한 강한 영향력을 나타냄

- **새로움**
   - 새로운 내용의 도입을 측정
   - 높은 값은 더 혁신적인 기여를 나타냄

- **커뮤니케이션 밀도**
   - 메시지당 정보 내용을 측정
   - 높은 값은 정보가 더 풍부한 메시지를 나타냄

결과는 지정된 출력 디렉토리에 CSV 파일로 저장됩니다.

```bibtex
@software{gca_analyzer,
  title = {GCA Analyzer: Group Conversation Analysis Tool},
  author = {Xiao, Jianjun},
  year = {2025},
  url = {https://github.com/etShaw-zh/gca_analyzer}
}
