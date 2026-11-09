# 교사 대시보드 데이터 명세 (화면 8 · 반 대시보드)

> **데모 범위 (2026-11-06)**: 발달 지표만 보여 준다.
> - 아래 필드 중 `emotions`, `observation`, `safety`(감정 흐름·AI 관찰·고백 감지)는 기본 출력에서 빠진다.
> - 이 필드들은 `StoryAnalyzer(include_psych=True)`일 때만 나온다. 근거 데이터 확보 후 재검토한다.
> - 데모 화면: https://claude.ai/artifact/77AoNuCstzDuTVPviUwARr · 분석 결과: https://claude.ai/artifact/NR4idEaXtbXZaXaxgLZG8e
>
> 새로 추가된 발달 필드: `development.growth_evidence`, `milestones`, `next_steps`, `refined` (아래 '발달 추가 필드' 표)
>
> `children[]` 형식: `{child_id, growth_up[], growth_down[], milestones}`
> - `growth_up`, `growth_down`: 6주 성장에서 '늘었어요/줄었어요'가 붙은 지표 이름
> - `milestones`: 처음 나온 표현 수
>
> 사이드바 배지는 이 세 값으로 정한다.

예시 파일: [`outputs/dashboard_week.json`](../outputs/dashboard_week.json) (가상 데이터 18명 × 6주)
생성 코드: [`scripts/story_metrics.py`](../scripts/story_metrics.py) `StoryAnalyzer.child_week_report()`

## 1. 분석에 필요한 입력 (앱 → 분석)

발화 1건 = 아이가 한 장면에서 한 말 (교사 보정 화면을 거친 문장)

| 필드 | 형식 | 설명 |
|---|---|---|
| `child_id` | 문자열 | 아이 ID |
| `date` | `YYYY-MM-DD` | 말한 날짜 |
| `book_id` | 문자열 | 그림책 ID |
| `scene` | 정수 | 장면 번호 |
| `text` | 문자열 | 아이가 한 말. 교사가 보정했다면 보정 후 문장 |
| `prompt` | 문자열 | 바로 앞 AI 질문. 감정 낱말을 질문에서 따라 말한 것은 세지 않기 위해 필요 |
| `used_choice` | bool | 막혔을 때 선택지 도움을 받았는지 |

## 2. 분석 결과 (분석 → 화면)

```
{
  "class":    { "name", "n_children", "week_start", "consent": {"given","total"}, "alert_rule", "synthetic" },
  "children": [ { "child_id", "has_observation", "has_safety" } ],   // 왼쪽 아이 목록
  "reports":  { "<child_id>": <아이 1명 · 1주 보고서> }
}
```

### 아이 1명 · 1주 보고서

| 필드 | 화면 위치 | 내용 |
|---|---|---|
| `child_id`, `week_start`, `week_end`, `books` | 상단 이름 줄 | "11월 2일 ~ 8일 · 그림책 2권" |
| `cards.scenes` `{value, last_week}` | 카드 1 · 만든 장면 | 이번 주 장면 수, 지난주 |
| `cards.self_scenes` `{value, total, choice_help}` | 카드 2 · 스스로 이어 말한 장면 | "9 / 12 · 선택지 도움 3번" |
| `cards.mean_eojeol` `{value, last_week}` | 카드 3 · 평균 문장 길이 | 어절. 소수 첫째 자리 |
| `cards.new_words` `{count, examples[3]}` | 카드 4 · 처음 쓴 낱말 | 이 아이의 이전 기록 전체에 없던 내용어. 동사·형용사는 '-다' 형태 |
| `emotions[]` `{name, this_week, last_week, increased}` | 이야기 속 감정 표현 | 순서 고정: 기쁨, 놀람, 무서움, 슬픔, 화남. **낱말 수**이며 기분 점수가 아님 |
| `speech_trend[6]` `{week_start, mean_eojeol}` | 말하기 변화 · 최근 6주 | 기록이 없는 주는 `null` (선을 끊어 표시) |
| `observation` (없으면 `null`) | AI 관찰 · 참고용 카드 | 아래 표 |
| `themes[]` `{word, count, character, increased}` | 자주 나온 인물·주제 | `increased`이면 진한 테두리 ↑. 등장인물(`character`)에는 ↑를 붙이지 않음 |
| `safety[]` `{level, category, reason, text, date, book_id, scene}` | **별도 영역** (아래 3번) | 고백 감지 결과 |
| `development` | 이야기 발달 영역 | 아래 표 |

### `development` (발달 지표)

| 필드 | 내용 |
|---|---|
| `six_week_growth[]` `{key, label, unit, help, value, baseline, change, p_value}` | **변화 판정은 여기서만.** 최근 3주(`value`)를 그 전 3주(`baseline`)와 비교. `change`: `up` / `down` / `null`. 각 기간 이야기 6편 이상일 때만 판정하고, 아니면 `p_value: null` ("기록이 더 필요해요") |
| `this_week_vs_4weeks[]` (같은 형식) | 이번 주 값과 지난 4주 값. **판정 없음** (`change`는 항상 `null`). 한 주에 이야기 2~3편으로는 통계적으로 판정할 수 없다 |
| `story_grammar` `{units: {배경, 시작 사건, 목표, 시도, 결과, 반응}, mean_score, complete_episodes, stories}` | 이번 주 이야기문법. `units`는 그 단위가 들어간 이야기 비율(%), `mean_score`는 이야기 한 편 평균(0~17), `complete_episodes`는 목표→시도→결과를 갖춘 에피소드 수 |
| `internal_states` `{by_type: {지각, 생리, 의식, 감정, 정신, 말하기}, examples}` | 마음을 나타내는 말(내적 상태 용어) 종류별 횟수와 예 |
| `connectives` `{by_type: {원인, 순서, 대조}, example}` | 이음말 종류별 횟수와 예문 1개 |
| `emotion_vocab` `{this_week[], cumulative, new[]}` | 이번 주에 쓴 감정 낱말, 지금까지 쓴 감정 낱말 수, 이번 주 처음 쓴 감정 낱말 |

### 발달 추가 필드

| 필드 | 화면 위치 | 내용 |
|---|---|---|
| `weekly` `{mean_eojeol, conn_rate, sg_score, ist_rate, self_ratio: [6주 값]}` | 6주 동안의 변화 (지표별 작은 추세선) | 6주 전 → 이번 주 순서. 그 주에 이야기가 없으면 `null` |
| `recent_3w` `{units{6단위 %}, ist{6종 횟수}, conn{원인·순서·대조 횟수}, sentences}` | 이야기 구조(이야기 길), 잇는 말, 마음을 나타내는 말, **우리 반** 탭의 히트맵 | 최근 3주 기준 |
| `growth_evidence` `{before, recent, note}` | 이야기가 이렇게 달라졌어요 | 그 전 3주와 최근 3주의 '중간 수준' 장면 `{text, date, complexity}`. 가장 잘한 장면이 아니라 과장하지 않는 대표 장면 |
| `milestones[]` `{skill, date, text, stories}` | 최근 3주에 처음 나온 표현 | 그 전에는 없었고 최근 3주에 이야기 2편 이상에서 나온 표현. `skill`: 원인 이음말 · 대조 이음말 · 목표 말하기 · 인용 · 꾸미는 절 · 생각을 나타내는 말 · 완전한 에피소드 |
| `next_steps[]` `{target, question, reason}` | 다음에 해 볼 질문 | 최근 3주 이야기에서 덜 나온 요소를 찾아 만든 질문. 최대 2개, 아이가 최근 가장 많이 쓴 등장인물 이름이 들어감. 'AI 질문에 추가' 버튼으로 앱의 AI 질문 설정에 넘길 수 있음 |
| `refined` `{recent_3w, before_3w}` | 세부 지표 | 각 기간 `{stories, episode_levels{나열·묘사, 불완전한 에피소드, 완전한 에피소드, 여러 에피소드}, best_level, intro_explicit_pct, complex_pct, quotes_per_story, pronouns_per_10_sent}`. **판정 없이 값만** 보여 줌 |

지표 `key`:
- `mean_eojeol`: 평균 문장 길이. **문장(절) 하나의 어절 수**로, 연구의 MLC-w에 해당한다. 카드와 6주 그래프도 같은 기준이다.
- `ndw50`: 다른 낱말 수 (낱말 50개 기준)
- `conn_rate`: 이음말 (회/문장)
- `sg_score`: 이야기문법 (점/17)
- `ist_rate`: 마음을 나타내는 말 (회/문장)
- `self_ratio`: 스스로 이어 말한 장면 (%)

이야기문법·마음을 나타내는 말의 근거는 [`research_sources.md`](research_sources.md)에 있다. ENNI, 한국 4~6세 이야기 회상 연구, MAIN 한국어판의 개념을 참고했고, 채점 규칙은 직접 만든 자동 채점이다. 평가 도구의 그림·채점지·문항은 쓰지 않는다.

판정: 이야기 단위 순열 검정, 유의수준 0.01 ÷ 지표 6개 ≈ 0.0017. **판정 표시는 드물게 나오는 것이 정상**이다.

화면 시안: https://claude.ai/artifact/77AoNuCstzDuTVPviUwARr (`scripts/build_prototype.py`로 다시 만들 수 있음)

### `observation`

| 필드 | 예 | 표시 |
|---|---|---|
| `title` | "최근 이야기에 무서움 표현이 자주 나왔어요" | 카드 제목 |
| `detail` | "최근 2주 동안 12번 나왔어요. 그 전 4주 동안은 1번이었어요." | 제목 아래 |
| `top_words` | ["혼자", "무섭다"] | 제목에 함께 써도 됨 |
| `evidence[2]` `{text, date, book_id, scene}` | "밤에 혼자 있어서 무서웠어요." · 11/5 장면 8 | 근거 문장 2개 |
| `suggestion` | "자유놀이 시간에 이 그림책을 함께 보며…" | 제안 |
| `disclaimer` | "AI는 아이가 한 말의 패턴만 보여 줘요…" | 카드 아래 고정 문구 |
| `emotion`, `this_week`, `baseline`, `baseline_weeks`, `p_value` | | 화면 표시 안 함 (기록용) |

버튼: "관찰 일지에 옮기기" / "이미 알고 있어요 · 숨기기". 교사의 선택은 기록해 두었다가 규칙 개선에 쓴다.

## 3. safety (고백 감지)는 감정 그래프와 분리

- `level`: `알림` (실제 어른·가족이 해를 입힘, 아이 본인의 죽음·자해 생각) / `확인 필요` (또래, 누가 했는지 불분명, 가벼운 체벌)
- `category`: 신체 폭력 / 성적 접촉 / 폭언·위협 / 죽음·자해 생각
- **점수나 그래프로 만들지 않는다.** 문장 원문, 날짜, 장면, `reason`만 보여 준다.
- 문구 예: "아이가 이렇게 말했어요. 확인이 필요할 수 있어요."
- 교사 선택지: "확인했어요" / "이야기 속 내용이에요" / "원장·전문가와 상의"
- 왼쪽 목록에는 `has_safety` 표시를 두되, 다른 교사나 보호자 화면에는 노출하지 않는다.

## 4. AI 관찰 규칙 (현재 값)

`class.alert_rule`에 같이 실린다. 근거는 [`alert_calibration.md`](alert_calibration.md)에 있다.

- 대상: 무서움·슬픔·화남이 **늘어날 때**
- 최근 2주를 그 전 4주와 비교한다. 기록이 2주 이상 있어야 판정한다.
- **낱말 수가 아니라 '그 감정이 나온 이야기 수'를 센다**(이항 검정). 아주 무서운 이야기 한 편은 1편으로만 센다.
- 다음을 모두 만족해야 한다.
  - **세기**: 최근 2주 횟수가 그 아이의 지난 4주에 비해 유의하게 많음 (음이항 검정, 유의수준 0.001, 4번 이상)
  - **퍼짐**: 최근 이야기의 60% 이상(3편 이상)에 나오고, 그 비율이 전보다 높음
- 무서운 이야기 한 편은 퍼짐에서, 원래 가끔 무서움을 말하던 아이는 세기에서 걸러진다.
- 모의실험 결과(이야기 속 감정 2회/100어절 가정): 오경보 0.6%, 3배 증가 감지 22%, 5배 증가 감지 49%.
- 한계: 이야기가 2주에 6편 정도면 '진짜 변화'와 '우연히 무서운 이야기가 몰린 경우'가 데이터로 구분되지 않을 때가 있다. 그래서 규칙을 보수적으로 잡았고, 강하고 여러 이야기에 걸친 변화만 알린다.
- 기준값은 모의실험 가정에 근거한다. 시범 데이터가 생기면 `calibrate_alerts.py`로 다시 정한다.

## 2026-11-09 추가: 나이별 또래 참고, 질문 두 단계, 형태소 문장 길이

**입력에 아이 나이가 필요하다.** `child_week_report(..., age_months=)`. 앱의 아이 정보에 생년월(또는 개월 수)을 저장해 넘겨 준다.

| 필드 | 내용 |
|---|---|
| `age_months` | 아이 나이(개월). 없으면 또래 참고를 보여 주지 않음 |
| `development.age_ref` | 또래 참고 `{age, 배경, 등장인물, 시작 사건, 목표, 시도, 결과, 반응(%), sentences_per_story, structure_8, mlc_m:[평균, 표준편차]}`. 4세(48~59개월)·5세(60~71개월) 값. 한유진·유안진(2001), 김기순 외(2007) |
| `development.recent_3w.sentences_per_story` | 최근 3주 그림책 한 권의 평균 문장 수 |
| `development.recent_3w.units.등장인물` | 이야기 요소에 '등장인물' 추가 (7개) |
| `development.next_steps[].tier` | `"먼저"`: 4~5세 대부분이 쓰는 요소(배경·등장인물·사건·시도·결과)가 또래보다 확실히 적을 때, 최대 2개 / `"한 단계 더"`: 목표·반응·생각·인물의 말. 앞의 요소가 갖춰졌을 때만 1개, "아직 쓰지 않아도 괜찮아요" 포함 |
| `cards.mlc_m` `{value, last_week}` | 문장 하나의 형태소 수 |
| 성장 판정 지표 key | `mean_eojeol` 대신 **`mlc_m`** (평균 문장 길이, 형태소). 어절 수는 `cards.mean_eojeol`로 계속 제공 |
