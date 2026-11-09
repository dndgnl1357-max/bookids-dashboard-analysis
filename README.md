# Bookids 교사 대시보드 분석

[Bookids](https://bookids-nine.vercel.app)는 유치원 아이(만 4~5세)가 그림을 보고 말하면 AI 토리가 "그다음엔 어떻게 될까?"를 되물어, 아이의 말이 모여 그림책이 되는 앱입니다.
이 저장소는 그 그림책 기록에서 **말하기와 이야기 짓기의 발달**을 읽어 내 보호자·선생님 화면에 보여 주는 분석 코드입니다.

- 분석 리포트: [reports/site/index.html](reports/site/index.html)
- 대시보드 데모: [reports/dashboard_prototype.html](reports/dashboard_prototype.html) (가상 반 18명 × 6주, 실제 아이의 말이 아님)
- 웹 연동용 데이터 명세: [reports/dashboard_spec.md](reports/dashboard_spec.md)
- 근거 자료 정리: [reports/research_sources.md](reports/research_sources.md)

## 무엇을 재나

| 지표 | 내용 | 표시 |
|---|---|---|
| 평균 문장 길이 | 문장 하나의 형태소 수 (화면에는 어절 수도) | 6주 성장 판정 |
| 잇는 말 | 그래서·그런데·-아서·-다가 | 6주 성장 판정 |
| 이야기 구성 | 배경·등장인물·시작 사건·목표·시도·결과·반응 (0~17점) | 6주 성장 판정 |
| 마음을 나타내는 말 | 지각·생리·의식·감정·정신·말하기 | 6주 성장 판정 |
| 스스로 이어 말한 장면 | 선택지 도움 없이 이어 말한 비율 | 6주 성장 판정 |
| 세부 지표 | 이야기 짜임 단계, 인물 소개, 이어진 문장, 인물의 말, 가리키는 말, 서로 다른 낱말 | 값만 |

- 성장 판정은 최근 3주와 그 전 3주의 그림책을 비교하는 순열 검정(유의수준 0.01 ÷ 지표 5개)으로, 확실할 때만 표시합니다.
- 4세·5세 또래 참고 범위와 '다음에 해 볼 질문'(먼저 해 볼 질문 / 한 단계 더)은 한국 연구(한유진·유안진 2001, 김기순 외 2007)를 따릅니다.
- 아이의 심리 상태 추정, 감정 변화 알림, 피해 고백 감지는 근거 데이터가 부족해 이번 데모에서 뺐습니다. 코드는 `StoryAnalyzer(include_psych=True)`로 남아 있습니다.

## 실행

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS·Linux는 .venv/bin/python

# 가상 반 데모: 기록 생성 → 지표 계산 → 대시보드 HTML
.venv/Scripts/python scripts/demo_class.py
.venv/Scripts/python scripts/build_prototype.py

# 자동 채점 점검 (이야기 구성 5개, 세부 지표 9개)
.venv/Scripts/python scripts/test_story_grammar.py
```

실제 앱 기록을 분석하려면 `story_metrics.StoryAnalyzer`에 발화 기록(아이 ID, 날짜, 그림책 ID, 장면, 아이 말, 바로 앞 토리 질문, 선택지 도움 여부)과 아이 나이(개월)를 넣습니다. 형식은 [dashboard_spec.md](reports/dashboard_spec.md)에 있습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `scripts/dev_metrics.py` | 발달 지표, 이야기 구성 자동 채점, 또래 참고, 다음 질문 |
| `scripts/story_metrics.py` | 발화 기록 → 아이 1명·1주 대시보드 데이터 |
| `scripts/demo_class.py` | 가상 반 기록 생성과 데모 실행 |
| `scripts/build_prototype.py` | 대시보드 데모 HTML (팀 앱 디자인) |
| `scripts/viz_report.py`, `viz_*.py` | 리포트 그림 |
| `lexicon/` | 감정 사전, 마음을 나타내는 말 사전 |
| `data/demo/` | 가상 반 발화 기록 |
| `outputs/` | 가상 반 대시보드 데이터 (JSON) |
| 그 밖의 `scripts/*.py` | 초기 탐색(상담 데이터 표 만들기, 감정 낱말, 예측 모델, 고백 감지, 음성 점검) |

## 상담 데이터에 대해

초기 탐색에는 AI Hub의 [아동·청소년 상담 데이터](https://aihub.or.kr)를 썼습니다. 재배포가 금지되어 있어 **원본, 추출한 표, 상담 문장을 인용한 보고서, 그 데이터로 학습한 모델은 이 저장소에 없습니다.**
AI Hub에서 직접 내려받아 저장소 바깥 `../024.아동·청소년 상담 데이터/`에 두면 `build_tables.py`부터 다시 실행할 수 있습니다.
점검 결과 이 데이터의 녹음은 연출일 가능성이 커서(상담사와 '아이' 목소리 높이 상관 0.96), 발달 지표의 근거로는 쓰지 않았습니다.
