"""고백 감지: 아이 말에서 '실제 사람에게 해를 입은 이야기'를 이야기 속 사건과 구분해 골라낸다.

점수가 아니라 문장 단위로 판정하고 근거를 함께 돌려준다. 교사에게 보여 줄 때도 문장 + 근거만 보여 준다.

    from disclosure import Detector
    det = Detector()
    det.detect("아빠가 술 마시고 나를 때렸어요.", context=["토끼가 놀러 갔어요."])
    -> [{"level": "알림", "category": "신체 폭력", "agent": "아빠", "victim": "나", ...}]

판정 단계
    알림     : 실제 어른·가족이 해를 입힌 말, 또는 아이 본인의 죽음·자해 생각
    확인 필요 : 해를 입힌 사람이 불분명하거나 또래(친구)인 경우
    (없음)   : 이야기 등장인물(늑대·공룡 등)의 사건, 부정된 말('안 때려요'), 아이가 가해자인 말
"""
from kiwipiepy import Kiwi

SELF = {"나", "저", "내", "제", "우리", "저희"}
REAL_ADULT = {
    "엄마", "아빠", "어머니", "아버지", "새엄마", "새아빠", "계모", "계부", "부모", "부모님",
    "할머니", "할아버지", "삼촌", "외삼촌", "이모", "고모", "이모부", "고모부", "숙모", "큰아빠", "큰엄마",
    "형", "오빠", "언니", "누나", "선생", "쌤", "담임", "보모", "교장", "원장", "코치", "목사", "기사",
    "아저씨", "아줌마", "어른", "어른들", "이웃", "사촌", "남자", "여자", "사람", "남자친구", "여자친구",
}
REAL_PEER = {"친구", "애", "애들", "동생", "반", "짝"}
CHARACTER = {
    "토끼", "늑대", "곰", "여우", "사자", "호랑이", "공룡", "티라노", "강아지", "개", "고양이", "다람쥐",
    "새", "돼지", "소", "말", "양", "염소", "닭", "병아리", "오리", "쥐", "뱀", "악어", "코끼리", "원숭이",
    "기린", "펭귄", "물고기", "상어", "고래", "거북이", "개구리", "나비", "벌", "개미", "괴물", "마녀",
    "용", "로봇", "왕자", "공주", "왕", "여왕", "유령", "귀신", "해적", "악당", "도깨비", "요정", "거인",
    "외계인", "슈퍼맨", "경찰", "도둑", "아기", "인형", "곰돌이", "토순이",
}
NOUN_TAGS = {"NNG", "NNP", "NP", "NNB"}
SUBJ = {"JKS", "JX"}                  # 가/이/께서, 는/은/도
DATIVE = {"한테", "에게", "에게서", "한테서", "께"}

HARM = {  # (형태소, 품사) -> 분류
    ("때리", "VV"): "신체 폭력", ("차", "VV"): "신체 폭력", ("꼬집", "VV"): "신체 폭력",
    ("밀치", "VV"): "신체 폭력", ("밀", "VV"): "신체 폭력", ("던지", "VV"): "신체 폭력",
    ("패", "VV"): "신체 폭력", ("할퀴", "VV"): "신체 폭력", ("죽이", "VV"): "신체 폭력",
    ("만지", "VV"): "성적 접촉",
    ("쫓아내", "VV"): "폭언·위협", ("가두", "VV"): "폭언·위협", ("버리", "VV"): "폭언·위협",
}
LEVEL_ORDER = {"알림": 2, "확인 필요": 1}
BODY = {"얼굴", "머리", "뺨", "볼", "몸", "팔", "다리", "등", "배", "가슴", "엉덩이", "손", "발",
        "코", "입", "눈", "귀", "목", "허벅지", "거기", "팬티", "바지"}
# 대상이 사람·몸이 아니면 해가 아닌 동사 ('공을 찼어요', '인형을 버렸어요', '물건을 던져요')
NEEDS_PERSON = {"차", "던지", "버리", "밀"}
# 관용 표현: (앞 명사, 동사)
IDIOMS = {("멍", "때리"), ("숨", "차"), ("배", "차"), ("속", "차")}

# 부정 범위: 해 동사 뒤로 이 품사들은 같은 절로 보고 계속 읽는다 ('때린 적은 한 번도 없어요')
NEG_CONTINUE = {"ETM", "NNB", "NNG", "NR", "NP", "MM", "MAG", "JX", "JKS", "JKO", "XSN", "VX", "EP", "XSV"}
NEG_EC = {"지", "게", "어", "아", "진", "지는", "지도", "지만은"}
NEGATORS = {("않", "VX"), ("않", "VV"), ("없", "VA"), ("아니", "VCN"), ("말", "VX"), ("안", "MAG"), ("못", "MAG")}
MILD = {"꿀밤"}  # '꿀밤 때려요': 가벼운 체벌. 알림 대신 확인 필요로 낮춘다


def negated(sent, j):
    """해 동사 sent[j]가 부정·금지됐는지. 앞의 '안/못', 같은 절 안 뒤쪽의 '없/않/아니/말/안 해'."""
    if any(t in {("안", "MAG"), ("못", "MAG")} for t in sent[max(0, j - 2):j]):
        return True
    rest = sent[j + 1:j + 12]
    for i, (f, g) in enumerate(rest):
        if (f, g) in NEGATORS:
            return True
        if g in NEG_CONTINUE or (g == "EC" and f in NEG_EC) or g in {"VV", "VA"} and f in {"하", "보", "해보"}:
            continue
        if (f, g) == ("고", "EC") and i + 1 < len(rest) and rest[i + 1][0] in {"그런", "그러"}:
            continue  # '때리고 그런 사람 아니에요'
        return False
    return False


def kind(run):
    """명사 묶음의 종류. 머리(마지막) 명사로 보되, '엄마 토끼'처럼 머리가 등장인물이면 등장인물,
    '엄마 남자친구'처럼 앞에 어른이 있으면 어른으로 본다."""
    if isinstance(run, str):
        run = [run]
    head = run[-1]
    if head in SELF:
        return "self"
    if head in CHARACTER:
        return "char"
    if head in REAL_ADULT or any(n in REAL_ADULT for n in run[:-1]):
        return "adult"
    if head in REAL_PEER:
        return "peer"
    return "other"


def phrases(toks):
    """명사 묶음과 뒤따르는 조사: [(명사 묶음, 조사 형태, 조사 품사, 끝 위치)].
    '엄마 토끼가' -> ['엄마', '토끼'], '선생님이' -> ['선생']."""
    out, run = [], []
    for i, (f, g) in enumerate(toks):
        if g in NOUN_TAGS:
            run.append(f)
        elif g == "XSN" and run:
            continue  # '님', '들'
        elif run:
            if g.startswith("J"):
                out.append((run, f, g, i))
            run = []
    return out


def sentences(toks):
    cur = []
    for t in toks:
        cur.append(t)
        if t[1] == "SF":
            yield cur
            cur = []
    if cur:
        yield cur


class Detector:
    def __init__(self, kiwi=None):
        self.kiwi = kiwi or Kiwi()

    def tok(self, text):
        return [(t.form, t.tag.split("-")[0]) for t in self.kiwi.tokenize(text)]

    def detect(self, text, context=(), toks=None, context_toks=None):
        """text: 판정할 말. context: 바로 앞의 말들 (이야기 앞 문장, 상담이라면 질문).
        toks/context_toks를 주면 형태소 분석을 건너뛴다."""
        toks = toks if toks is not None else self.tok(text)
        ctx = context_toks if context_toks is not None else [self.tok(c) for c in context]
        history = [s for c in ctx for s in sentences(c)]
        found = []
        for sent in sentences(toks):
            r = self._sentence(sent, history)
            if r:
                found.append({**r, "sentence": "".join(f if g != "SF" else f for f, g in sent)})
            history.append(sent)
        return found

    def _last_subject(self, history):
        """앞 문맥에서 가장 최근의 사람·등장인물 주어 (이야기 속 '그'가 누구인지)."""
        for sent in reversed(history[-3:]):
            subj = [p for p in phrases(sent) if p[2] in SUBJ and kind(p[0]) != "other"]
            if subj:
                return subj[-1][0]
        return None

    def _sentence(self, sent, history):
        ph = phrases(sent)
        nouns = {f for f, g in sent if g in NOUN_TAGS}
        mentions_char = any(kind(p[0]) == "char" for p in ph) or any(
            kind(p[0]) == "char" for s in history[-2:] for p in phrases(s))

        for j, t in enumerate(sent):
            cat = HARM.get(t)
            passive = False
            if t == ("맞", "VV"):  # '맞았어요'(맞음) vs '맞아요'(동의)
                before = sent[max(0, j - 3):j]
                if any(f in DATIVE for f, g in before) or ("매", "NNG") in before or \
                        ("었", "EP") in sent[j + 1:j + 2]:
                    cat, passive = "신체 폭력", True
            if t == ("욕", "NNG") and j + 1 < len(sent) and sent[j + 1][1] == "XSV":
                cat = "폭언·위협"
            neg_at = j
            if t == ("죽", "VV") and (("싶", "VX") in sent[j + 1:j + 4] or ("버리", "VX") in sent[j + 1:j + 4]):
                cat = "죽음·자해 생각"
                # '죽고 싶은 생각은 안 해봤어요': 부정은 '싶' 뒤에 온다
                neg_at = next(k for k in range(j + 1, j + 4) if sent[k] in {("싶", "VX"), ("버리", "VX")})
            if t == ("자살", "NNG"):
                cat = "죽음·자해 생각"
            if not cat or negated(sent, neg_at):
                continue
            if j > 0 and (sent[j - 1][0], t[0]) in IDIOMS or \
                    j > 1 and (sent[j - 2][0], t[0]) in IDIOMS:  # '멍때리다', '숨이 차다'
                continue

            before = [p for p in ph if p[3] < j]
            subj = [p[0] for p in before if p[2] in SUBJ]
            obj = [p[0] for p in before if p[2] == "JKO" or (p[2] == "JKB" and p[1] in DATIVE)
                   or (p[2] == "JKG" and kind(p[0]) == "self")]

            if cat == "죽음·자해 생각":
                who = subj[-1] if subj else None
                if who is None or kind(who) == "self":
                    return self._result("알림", cat, None, who or "(생략: 아이 본인)", "죽음·자해를 바라는 말")
                continue

            if t[0] in NEEDS_PERSON and not self._targets_person(sent, j, nouns):
                continue  # 대상이 물건이면 해가 아님 ('공을 찼어요', '제 인형을 버렸어요')

            if passive:  # '아빠한테 맞았어요': 한테 앞이 가해자, 주어(또는 생략)가 피해자
                dat = [p[0] for p in before if p[2] == "JKB" and p[1] in DATIVE]
                agent = dat[-1] if dat else None
                victim = subj[-1] if subj else None
            else:
                agent = subj[-1] if subj else None
                victim = obj[-1] if obj else None
            if not passive and (agent is None or kind(agent) == "other"):
                # 주어가 없거나 사람이 아니면('기분이 안 좋아서 때린') 앞 문맥의 사람 주어를 쓴다
                agent = self._last_subject(history) or agent
            ak, vk = kind(agent) if agent else None, kind(victim) if victim else None

            if ak == "char" or (vk == "char" and ak not in {"adult", "peer"}):
                continue  # 이야기 등장인물의 사건
            if ak == "self":
                continue  # 아이가 가해자인 말은 이 경로의 대상이 아님
            if ak == "adult" and nouns & MILD:
                return self._result("확인 필요", cat, agent, victim, "어른의 가벼운 체벌 (꿀밤)")
            if ak == "adult":
                return self._result("알림", cat, agent, victim, "실제 어른·가족이 해를 입힘")
            if ak == "peer":
                return self._result("확인 필요", cat, agent, victim, "또래가 해를 입힘")
            if vk == "self":
                return self._result("확인 필요", cat, agent, victim, "아이가 해를 입었으나 누가 했는지 불분명")
            if not mentions_char:
                return self._result("확인 필요", cat, agent, victim, "누가 누구에게 했는지 불분명")
        return None

    @staticmethod
    def _targets_person(sent, j, nouns):
        """차다·던지다·버리다·밀다의 대상이 사람(또는 몸)인지.
        주어('엄마가')와 소유('제 인형')는 대상이 아니다. 대상이 아예 안 나오면 사람으로 본다."""
        if nouns & BODY:
            return True
        has_thing = False
        run = []
        for f, g in sent[:j] + [("", "END")]:
            if g in NOUN_TAGS:
                run.append(f)
                continue
            if g == "XSN" and run:
                continue
            if run and not (g in SUBJ or g == "JKG"):  # 주어('담임 선생님이')와 소유('제 인형'의 '제')는 대상 아님
                k = kind(run)
                if k in {"self", "adult", "peer"}:
                    return True
                if k in {"other", "char"} and run[-1] not in {"때", "매일", "지난주", "어제", "오늘", "그때", "거", "것"}:
                    has_thing = True
            run = []
        return not has_thing

    @staticmethod
    def _result(level, cat, agent, victim, reason):
        name = lambda r: " ".join(r) if isinstance(r, list) else r
        return {"level": level, "category": cat, "agent": name(agent), "victim": name(victim), "reason": reason}


def summarize(found):
    """여러 문장 결과 중 가장 높은 단계."""
    if not found:
        return None
    return max(found, key=lambda r: LEVEL_ORDER[r["level"]])["level"]
