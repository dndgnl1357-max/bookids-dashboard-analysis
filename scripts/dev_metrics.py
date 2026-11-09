"""발달 지표: 아이 이야기의 어휘·문장·이음말·이야기문법·내적 상태 용어·감정 어휘.

근거 (reports/research_sources.md)
- 이야기문법: 그림 이야기 평가에서 쓰는 이야기문법 틀(배경 · 시작 사건 · 목표 · 시도 · 결과 · 반응,
  목표-시도-결과가 갖춰진 '완전한 에피소드'). ENNI(Schneider 2006)에서 4~9세 나이에 따라 증가하고
  언어장애 판별 정확도 80.8%. 한국 4~6세 이야기 회상 연구(언어청각장애연구 12(1), 2007)에서도 이야기문법
  총점과 절 단위 평균 길이(MLC)가 나이에 따라 증가.
- 내적 상태 용어: 인물의 마음을 말하는 낱말 6종 (지각·생리·의식·감정·정신·말하기).
- 아래 규칙은 그 개념을 참고해 직접 만든 자동 채점이다. 평가 도구(MAIN 등)의 그림·채점지·문항은 쓰지 않는다.

story_metrics.StoryAnalyzer.analyze() 결과(발화마다 형태소 'toks' 포함)를 받아 계산한다.
변화 표시(↑↓)는 '우연히 흔들리는 범위'를 넘을 때만 붙인다:
그림책(이야기) 단위로 이번 주와 지난 4주를 섞어 다시 나누는 순열 검정으로 p값을 구한다.
발화 단위가 아니라 이야기 단위로 섞는 이유는, 같은 이야기 안의 발화끼리는 비슷해서
발화 단위로 보면 변화를 실제보다 확실하게 착각하기 때문이다.
"""
from math import lgamma, exp
from pathlib import Path

import numpy as np
import pandas as pd

from disclosure import CHARACTER, REAL_ADULT, REAL_PEER

_IST = pd.read_csv(Path(__file__).resolve().parents[1] / "lexicon" / "ist_lexicon.csv")
IST = {(r.형태소, r.품사): r.종류 for r in _IST.itertuples()}
IST_NEEDS_HA = {"생각", "결정", "기억", "계획", "말", "대답", "경고"}  # 뒤에 '하다'가 올 때만
IST_TYPES = ["지각", "생리", "의식", "감정", "정신", "말하기"]

# 이음말 (사건을 원인·순서·대조로 잇는 말)
CONNECTIVES = {
    "원인": {("그래서", "MAJ"), ("그러니까", "MAJ"), ("왜냐하면", "MAG"), ("어서", "EC"), ("아서", "EC"),
            ("서", "EC"), ("니까", "EC"), ("으니까", "EC")},
    "순서": {("그리고", "MAJ"), ("다가", "EC"), ("결국", "MAG"), ("마침내", "MAG"), ("드디어", "MAG"),
            ("나중", "NNG"), ("다음", "NNG")},
    "대조": {("그런데", "MAJ"), ("하지만", "MAJ"), ("그래도", "MAJ"), ("그러나", "MAJ"),
            ("는데", "EC"), ("은데", "EC"), ("ㄴ데", "EC"), ("지만", "EC")},
}
CONN_LOOKUP = {t: k for k, ts in CONNECTIVES.items() for t in ts}

PLACES = {"숲", "숲속", "바다", "집", "공원", "학교", "유치원", "하늘", "동굴", "놀이터", "산", "강", "마을", "성",
          "방", "우주", "기차역", "나라", "섬", "정글", "농장", "동물원", "시장", "들판", "호수", "호숫가"}
TIMES = {"밤", "아침", "저녁", "옛날", "낮", "겨울", "여름", "봄", "가을"}
# 시작 사건: 이야기를 움직이는 일 (문제, 갑작스러운 일, 인물이 무언가를 봄·느낌)
INITIATING = {"잃어버리", "잃", "없어지", "다치", "넘어지", "싸우", "갇히", "부서지", "망가지", "떨어지", "괴물",
              "늦", "갑자기", "나타나"}
SUCCESS = {"찾", "잡", "구하", "고치", "도와주", "구해주", "치료", "화해", "만나", "성공", "도착하", "이기"}
OUTCOME_MARK = {"드디어", "결국", "마침내"}
ATTEMPT_MARK = {"열심히", "노력", "애쓰", "찾아보", "찾아다니"}
SG_UNITS = ["배경", "등장인물", "시작 사건", "목표", "시도", "결과", "반응"]

# 또래 참고 범위: 한유진·유안진(2001) 3~5세가 새 이야기를 지어 말한 연구, 나이별 20명.
# 각 요소가 들어간 이야기 비율. '목표'는 연구의 '내적 상태'(바람·목표), '시작 사건'은 '계기가 되는 사건'.
AGE_REF = {
    4: {"배경": 55, "등장인물": 70, "시작 사건": 70, "목표": 20, "시도": 65, "결과": 60, "반응": 10,
        "sentences_per_story": 7.9, "structure_8": 3.80},
    5: {"배경": 85, "등장인물": 95, "시작 사건": 95, "목표": 25, "시도": 100, "결과": 80, "반응": 11,
        "sentences_per_story": 14.2, "structure_8": 5.41},
}
# 김기순 외(2007) 이야기 다시 말하기: 형태소로 본 평균절길이(MLC-m). 들은 이야기를 다시 말하는 과제라
# 스스로 지어 말할 때보다 높게 나온다. 방향을 보는 참고값으로만 쓴다.
MLC_M_REF = {4: (15.05, 2.07), 5: (16.96, 2.05), 6: (19.03, 2.70)}
# 4~5세 대부분이 쓰는 요소 / 아직 드문 것이 정상인 요소 (2001 연구에서 5세 사용 비율 80% 이상 vs 25% 이하)
CORE_UNITS = ["배경", "등장인물", "시작 사건", "시도", "결과"]
STRETCH_UNITS = ["목표", "반응"]


def age_group(age_months):
    if age_months is None:
        return None
    return 4 if age_months < 60 else 5 if age_months < 72 else 6
ACTION_TAGS = {"VV"}

CONTENT = {"NNG", "NNP", "VV", "VA"}
STOP = {"것", "거", "때", "수", "등", "중", "번", "개", "하", "있", "없", "되", "이", "같", "그렇", "그러",
        "주", "들", "놓", "두", "알", "싶", "안", "뭐", "날"}


# ---- 발화·이야기 단위 계산 --------------------------------------------------------------

def connectives(toks):
    return [CONN_LOOKUP[t] for t in toks if t in CONN_LOOKUP]


def content_lemmas(toks):
    return [f for f, g in toks if g in CONTENT and f not in STOP]


def sentences(toks):
    """형태소 목록 -> 문장(마침표·물음표 등 SF 기준) 목록. 절 단위(C-unit)의 근사."""
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if t[1] == "SF":
            out.append(cur)
            cur = []
    if any(g not in {"SF", "SP", "SS"} for f, g in cur):
        out.append(cur)
    return out


def ist_terms(toks):
    """내적 상태 용어 [(종류, 낱말)]. '보다'는 본동사(VV)만, '생각·말' 같은 명사는 뒤에 '하다'가 올 때만."""
    out = []
    for j, t in enumerate(toks):
        kind = IST.get(t)
        if not kind:
            continue
        if t[0] in IST_NEEDS_HA and not (j + 1 < len(toks) and toks[j + 1][0] == "하"):
            continue
        out.append((kind, t[0]))
    return out


# ---- 세부 지표 (언어 연구의 미시구조 지표를 참고) ----------------------------------------------

PRONOUNS = {"걔", "얘", "쟤", "그거", "이거", "저거", "그것", "이것", "저것", "거기"}
QUOTE_EC = {"자고", "다고", "냐고", "라고", "으라고", "래"}
LINK_EC = {"어서", "아서", "서", "니까", "으니까", "다가", "는데", "은데", "ㄴ데", "지만", "면", "으면",
           "려고", "으려고", "도록", "면서", "으면서"}


def quotes(toks):
    """인용: 직접('같이 놀자'라고) + 간접('놀자고 했어요')."""
    direct = sum(t == ("라고", "JKQ") for t in toks)
    indirect = sum(toks[i][1] == "EC" and toks[i][0] in QUOTE_EC and toks[i + 1][0] in {"하", "말하", "말"}
                   for i in range(len(toks) - 1))
    return direct + indirect


def is_complex(sent):
    """복문: 절을 잇는 어미(-아서, -는데, -려고…)나 인용, 또는 주어·목적어를 가진 꾸미는 절('배가 고픈 토끼')."""
    if any(g == "EC" and f in LINK_EC for f, g in sent) or any(g == "JKQ" for f, g in sent):
        return True
    return any(sent[i][1] == "ETM" and any(g in {"JKS", "JKO"} for f, g in sent[max(0, i - 3):i])
               for i in range(len(sent)))


def pronoun_count(toks):
    return sum(g == "NP" and f in PRONOUNS for f, g in toks)


def intro_explicit(book_toks):
    """이야기 첫 문장에서 인물을 이름(명사)으로 소개했는지 ('토끼가…' vs '걔가…' / 주어 없음)."""
    sents = [s for toks in book_toks for s in sentences(toks)]
    if not sents:
        return False
    first = sents[0]
    return any(first[i][1] in {"NNG", "NNP"} and first[i + 1][1] in {"JKS", "JX"} and first[i][0] not in TIMES
               for i in range(len(first) - 1))


EPISODE_LEVELS = ["나열·묘사", "불완전한 에피소드", "완전한 에피소드", "여러 에피소드"]


def episode_level(grammar):
    """이야기 하나의 구조 단계 (그림 이야기 평가의 구조적 복합성 개념 참고)."""
    if grammar["episodes"] == 0:
        return 0
    if grammar["complete_episodes"] == 0:
        return 1
    if grammar["complete_episodes"] == 1 and grammar["episodes"] == 1:
        return 2
    return 3


# 처음 해낸 것: 발화 하나에서 확인할 수 있는 기술
SKILLS = ["원인 이음말", "대조 이음말", "목표 말하기", "인용", "꾸미는 절", "생각을 나타내는 말"]


def skills(toks):
    found = set()
    conn = connectives(toks)
    if "원인" in conn:
        found.add("원인 이음말")
    if "대조" in conn:
        found.add("대조 이음말")
    sents = sentences(toks)
    if any("goal" in _units(s) for s in sents):
        found.add("목표 말하기")
    if quotes(toks):
        found.add("인용")
    if any(s[i][1] == "ETM" and any(g in {"JKS", "JKO"} for f, g in s[max(0, i - 3):i])
           for s in sents for i in range(len(s))):
        found.add("꾸미는 절")
    if any(k == "정신" for k, _ in ist_terms(toks)):
        found.add("생각을 나타내는 말")
    return found


def complexity(toks, eojeol):
    """발화 하나의 복잡도 (근거 문장 고르기용): 어절 + 이음말·복문·마음 말·인용 가중."""
    sents = sentences(toks)
    return eojeol + 2 * len(connectives(toks)) + 2 * sum(is_complex(s) for s in sents) \
        + len(ist_terms(toks)) + 2 * quotes(toks)


# 다음 단계 질문: 아이가 아직 잘 안 하는 것을 이끌어 내는 질문. 이야기문법이 발달하는 대략의 순서로 본다
# (배경·사건 → 결과 → 시도 → 목표 → 반응). 설계용 규칙이며 효과는 시범 운영에서 확인해야 한다.
NEXT_STEPS = [
    ("배경", "언제, 어디에서 있었던 일일까?", "이야기의 시간·장소를 말한 경우가 적어요"),
    ("등장인물", "이 이야기에는 누가 나와?", "누가 나오는 이야기인지 말한 경우가 적어요"),
    ("시작 사건", "그런데 {c}에게 무슨 일이 생겼을까?", "이야기를 움직이는 사건이 드물어요"),
    ("결과", "그래서 어떻게 됐을까?", "일이 어떻게 끝났는지 말한 경우가 적어요"),
    ("시도", "그래서 {c}{neun} 어떻게 했을까?", "인물이 문제를 풀려고 한 행동이 드물어요"),
    ("목표", "{c}{neun} 무엇을 하고 싶었을까?", "인물이 무엇을 원했는지는 4~5세에 아직 드물게 말해요"),
    ("반응", "그때 {c}{neun} 기분이 어땠을까?", "일이 끝난 뒤 인물의 마음은 5세도 대부분 말하지 않아요"),
]
EXTRA_STEPS = [
    ("원인 이음말", "{c}{neun} 왜 그랬을까?", "'그래서·-아서'처럼 이유를 잇는 말이 아직 없어요"),
    ("인용", "{c}{neun} 친구한테 뭐라고 말했을까?", "인물의 말을 옮기는 표현이 아직 없어요"),
    ("생각을 나타내는 말", "{c}{neun} 무슨 생각을 했을까?", "'생각하다·알다'처럼 생각을 말하는 표현이 아직 없어요"),
]


def _josa(word, pair):
    return pair[0] if (ord(word[-1]) - 0xAC00) % 28 != 0 else pair[1]


def next_steps(books, recent_skills, character, age_months=None, max_n=2):
    """최근 이야기에서 덜 나온 요소를 찾아 교사·토리가 할 질문을 제안. 두 단계로 나눈다.
    - 먼저 해 볼 질문(core): 4~5세 대부분이 쓰는 요소(배경·등장인물·사건·시도·결과)가 또래보다 적을 때
    - 한 단계 더(stretch): 목표·반응·생각·인용처럼 4~5세에 아직 드문 것이 정상인 요소.
      core가 다 갖춰졌을 때만 하나 제안하고, '아직 쓰지 않아도 괜찮아요'를 함께 보여 준다.
    근거: 한유진·유안진(2001)에서 내적 상태 20~25%, 반응 10~11%로 4~5세에 나이 차이 없이 드묾."""
    if not books:
        return []
    fill = lambda q: q.format(c=character, neun=_josa(character, ("은", "는")))
    ref = AGE_REF.get(age_group(age_months)) or AGE_REF[4]
    rate = {u: float(np.mean([b["grammar"]["units"][u] for b in books])) for u in SG_UNITS}
    core = []
    for unit, q, why in NEXT_STEPS:
        if unit in CORE_UNITS and rate[unit] * 100 < min(ref[unit], 80) - 10:  # 또래보다 확실히 적을 때
            core.append({"tier": "먼저", "target": unit, "question": fill(q),
                         "reason": f"{why} (최근 그림책 {len(books)}권 중 {round(rate[unit] * len(books))}권, "
                                   f"또래는 보통 {ref[unit]}%)"})
    if core:
        return core[:max_n]
    stretch = []
    for unit, q, why in NEXT_STEPS:
        if unit in STRETCH_UNITS and rate[unit] < 0.5:
            stretch.append({"tier": "한 단계 더", "target": unit, "question": fill(q),
                            "reason": f"{why}. 아직 쓰지 않아도 괜찮아요"})
    for skill, q, why in EXTRA_STEPS:
        if skill not in recent_skills:
            # 생각·믿음(생각하다·알다)은 바람·감정 이해 뒤에 발달한다(마음이론 연구).
            # 아직 목표(바람)나 감정을 말하지 않는 아이에게는 생각 질문을 미룬다
            if skill == "생각을 나타내는 말" and not ({"목표 말하기"} & recent_skills or rate["반응"] > 0):
                continue
            stretch.append({"tier": "한 단계 더", "target": skill, "question": fill(q),
                            "reason": f"{why}. 아직 쓰지 않아도 괜찮아요"})
    return stretch[:1]


def _units(sent):
    """문장 하나에 들어 있는 이야기문법 단위 후보."""
    forms = [f for f, g in sent]
    fs = set(forms)
    u = set()
    if fs & TIMES or ("어느", "MM") in sent:
        u.add("time")
    if any(sent[i][0] in PLACES and i + 1 < len(sent) and sent[i + 1][0] in {"에", "에서"} for i in range(len(sent))):
        u.add("place")
    if fs & INITIATING or any(k in {"생리", "지각"} for k, _ in ist_terms(sent)):
        u.add("ie")
    if any(t in {("려고", "EC"), ("으려고", "EC")} for t in sent) or \
            any(sent[i] == ("고", "EC") and sent[i + 1] == ("싶", "VX") for i in range(len(sent) - 1)) or \
            any(sent[i] == ("기", "ETN") and sent[i + 1][0] == "로" for i in range(len(sent) - 1)) or \
            fs & {"원하", "마음먹", "결심"}:
        u.add("goal")
    if fs & ATTEMPT_MARK or any(sent[i] == ("어", "EC") and sent[i + 1] == ("보", "VX") for i in range(len(sent) - 1)):
        u.add("attempt_mark")
    if any(g in ACTION_TAGS for f, g in sent):
        u.add("action")
    neg = ("못", "MAG") in sent
    if fs & OUTCOME_MARK or (fs & SUCCESS and any(g == "EP" for f, g in sent)) or (neg and "action" in u):
        u.add("outcome")
    if any(k == "감정" for k, _ in ist_terms(sent)):
        u.add("emotion")
    return u


def _subjects(sent):
    """주어 자리(가/이/는/은 앞)의 명사."""
    return [sent[i][0] for i in range(len(sent) - 1)
            if sent[i][1] in {"NNG", "NNP"} and sent[i + 1][1] in {"JKS", "JX"}]


def kind_char(noun):
    return noun in CHARACTER or noun in REAL_ADULT or noun in REAL_PEER


def mlc_m(sent):
    """문장 하나의 형태소 수 (문장부호 제외). 김기순 외(2007)가 권한 형태소 단위 문장 길이."""
    return sum(1 for f, g in sent if not g.startswith("S"))


def story_grammar(book_toks):
    """한 이야기(장면 순서대로의 형태소 목록들)의 이야기문법 자동 채점.
    배경 0~2점(시간·장소) + 에피소드마다 시작 사건·목표·시도·결과·반응 각 1점 (최대 3개 에피소드, 총 17점).
    완전한 에피소드 = 목표 → 시도 → 결과가 차례로 나온 것."""
    sents = [s for toks in book_toks for s in sentences(toks)]
    us = [_units(s) for s in sents]
    setting = int(any("time" in u for u in us[:3])) + int(any("place" in u for u in us[:3]))
    episodes, cur = [], None
    for u in us:
        opens = "ie" in u or "goal" in u
        if opens and (cur is None or cur["결과"]):
            if cur:
                episodes.append(cur)
            cur = {"시작 사건": False, "목표": False, "시도": False, "결과": False, "반응": False}
        if cur is None:
            continue
        if "ie" in u and not cur["목표"]:
            cur["시작 사건"] = True
        if "goal" in u:
            cur["목표"] = True
        elif (cur["목표"] or cur["시작 사건"]) and not cur["결과"] and \
                ("attempt_mark" in u or ("action" in u and "outcome" not in u)):
            cur["시도"] = True
        # 목표를 말한 문장('만나고 싶었어요')은 결과가 아니다
        if "outcome" in u and "goal" not in u and (cur["시도"] or cur["목표"] or cur["시작 사건"]):
            cur["결과"] = True
        elif cur["결과"] and "emotion" in u:
            cur["반응"] = True
    if cur:
        episodes.append(cur)
    episodes = episodes[:3]
    score = setting + sum(sum(e.values()) for e in episodes)
    complete = sum(e["목표"] and e["시도"] and e["결과"] for e in episodes)
    # 등장인물: 인물(동물·사람)이 주어로 나와 이야기를 이끄는지. 점수(0~17)에는 넣지 않고 요소로만 본다
    has_char = any(kind_char(p) for s in sents for p in _subjects(s))
    units = {"배경": setting > 0, "등장인물": has_char,
             **{k: any(e[k] for e in episodes) for k in ["시작 사건", "목표", "시도", "결과", "반응"]}}
    return {"score": score, "complete_episodes": complete, "episodes": len(episodes), "units": units}


def rarefied_ndw(lemmas, n=50):
    """다른 낱말 수를 표본 크기에 공정하게: 낱말 n개를 뽑았을 때 기대되는 서로 다른 낱말 수 (희박화).
    말을 많이 한 주가 그냥 많이 말해서 높게 나오는 것을 막는다. 낱말이 n개보다 적으면 None."""
    N = len(lemmas)
    if N < n:
        return None
    counts = {}
    for w in lemmas:
        counts[w] = counts.get(w, 0) + 1
    lc = lambda a, b: lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)
    total = lc(N, n)
    return float(sum(1 - (exp(lc(N - k, n) - total) if N - k >= n else 0.0) for k in counts.values()))


# ---- 주 단위 지표: 이야기(book) 목록 -> 값 ----------------------------------------------------

def _utts(books):
    return [u for b in books for u in b["utts"]]


def _sent_lens(bs):
    return [n for u in _utts(bs) for n in u["sent_eojeol"]]


def _morph_lens(bs):
    return [n for u in _utts(bs) for n in u["sent_morph"]]


METRICS = {
    # key: (화면 이름, 단위, 계산 함수(이야기 목록 -> 값), 설명)
    # 문장 길이는 형태소 단위로 판정한다: 김기순 외(2007)에서 낱말 단위는 4세와 5세 차이가 없었고
    # 형태소 단위(조사·어미 포함)는 나이마다 달랐다. 화면 카드에는 읽기 쉬운 어절 수도 함께 보여 준다
    "mlc_m": ("평균 문장 길이", "형태소",
              lambda bs: float(np.mean(_morph_lens(bs))) if _morph_lens(bs) else None,
              "문장 하나의 평균 형태소 수(조사·어미 포함). 한국어 연구에서 4세와 5세 차이를 더 잘 보여 준 단위"),
    "conn_rate": ("이음말", "회 / 문장",
                  lambda bs: float(np.mean([len(u["conn"]) for u in _utts(bs)])) if _utts(bs) else None,
                  "'그래서·그런데·-아서·-다가'처럼 사건을 잇는 말이 한 문장에 몇 번 나오는지"),
    "sg_score": ("이야기문법", "점 / 17",
                 lambda bs: float(np.mean([b["grammar"]["score"] for b in bs])) if bs else None,
                 "배경·시작 사건·목표·시도·결과·반응을 얼마나 갖춰 말했는지 (이야기 한 편 평균)"),
    "ist_rate": ("마음을 나타내는 말", "회 / 문장",
                 lambda bs: float(np.mean([len(u["ist"]) / max(len(u["sent_eojeol"]), 1) for u in _utts(bs)]))
                 if _utts(bs) else None,
                 "보다·배고프다·기쁘다·생각하다·말하다처럼 인물의 마음과 상태를 말하는 낱말"),
    "self_ratio": ("스스로 이어 말한 장면", "%",
                   lambda bs: float(100 * np.mean([not u["used_choice"] for u in _utts(bs)])) if _utts(bs) else None,
                   "선택지 도움 없이 이어 말한 장면 비율"),
}


# 다른 낱말 수는 성장 판정에서 뺐다 (값만 보여 줌). 유아 어휘 다양도 측정 연구(Yang 외, 2022)에서
# 고정 길이 NDW는 나이에 따라 늘지만 집단 구분력이 거의 없고, 가장 나은 VocD도 이야기 맥락(소재)에 흔들렸다.
# Bookids는 아이가 매번 다른 소재로 이야기를 지어서 이 흔들림이 크다.
def ndw50(books):
    return rarefied_ndw([w for u in _utts(books) for w in u["lemmas"]])


def perm_test(books_a, books_b, f, n_perm=2000, rng=None):
    """이야기 단위 순열 검정. 반환: (a값, b값, 양측 p값)."""
    rng = rng or np.random.default_rng(0)
    va, vb = f(books_a), f(books_b)
    if va is None or vb is None:
        return va, vb, None
    obs = va - vb
    pool, na = books_a + books_b, len(books_a)
    hits, valid = 0, 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pool))
        a, b = f([pool[i] for i in idx[:na]]), f([pool[i] for i in idx[na:]])
        if a is None or b is None:
            continue
        valid += 1
        hits += abs(a - b) >= abs(obs) - 1e-12
    return va, vb, (hits + 1) / (valid + 1)


def compare(books_a, books_b, alpha=0.01 / len(METRICS), min_a=6, min_b=6, test=True):
    # alpha = 0.01 / 지표 수 (아이 한 명에게 지표 여러 개를 한꺼번에 보므로 우연한 ↑↓를 막는다)
    """지표마다 a(이번 기간) vs b(비교 기간). 변화 표시는 p < alpha이고 이야기 수가 충분할 때만.
    이야기가 적으면 순열 검정이 원리적으로 작은 p값을 낼 수 없다 (2권 vs 10권이면 최소 p = 1/66).
    그래서 주 단위 비교는 test=False로 값만 내고, 변화 판정은 3주 vs 3주(각 6권 이상)에서만 한다."""
    out = []
    for key, (label, unit, f, help_) in METRICS.items():
        enough = test and len(books_a) >= min_a and len(books_b) >= min_b
        va, vb, p = perm_test(books_a, books_b, f) if enough else (f(books_a), f(books_b), None)
        change = None
        if p is not None and p < alpha:
            change = "up" if va > vb else "down"
        out.append({"key": key, "label": label, "unit": unit, "help": help_,
                    "value": None if va is None else round(va, 2),
                    "baseline": None if vb is None else round(vb, 2),
                    "change": change, "p_value": None if p is None else float(f"{p:.2g}")})
    return out
