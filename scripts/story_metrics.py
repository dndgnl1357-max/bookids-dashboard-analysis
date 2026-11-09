"""Bookids 이야기 기록 -> 교사 대시보드 수치.

입력: 발화 기록 DataFrame (한 행 = 아이가 한 장면에서 한 말)
    child_id    아이 ID
    date        'YYYY-MM-DD'
    book_id     그림책 ID
    scene       장면 번호
    text        아이가 한 말 (교사 보정 후 문장)
    prompt      바로 앞 AI 질문 (없으면 빈 문자열). 질문의 감정 낱말을 따라 말한 것은 세지 않는다
    used_choice 막혔을 때 선택지 도움을 받았는지 (bool)

출력: child_week_report() -> 대시보드 한 화면(아이 1명, 1주)에 해당하는 dict. 형식은 reports/dashboard_spec.md

원칙
- 감정 수치는 '이야기에 나온 감정 낱말 수'이지 아이의 기분이 아니다.
- AI 관찰은 다른 아이가 아니라 그 아이의 지난 4주와 비교한다.
- 한 권의 무서운 그림책 때문에 울리지 않도록, 늘어난 감정이 2권 이상에 걸쳐 나와야 한다.
- 고백 감지(disclosure.py) 결과는 감정 그래프와 섞지 않고 safety로 따로 낸다.
"""
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, nbinom

import dev_metrics as dev
from disclosure import CHARACTER, Detector
from features import is_negated, is_wish

ROOT = Path(__file__).resolve().parents[1]
LEX = ROOT / "lexicon"

DASHBOARD_EMOTIONS = ["기쁨", "놀람", "무서움", "슬픔", "화남"]
WATCH_EMOTIONS = ["무서움", "슬픔", "화남"]   # AI 관찰 대상 (늘어날 때)
CONTENT_TAGS = {"NNG", "NNP", "VV", "VA"}
STOP = {"것", "거", "때", "수", "등", "중", "번", "개", "하", "있", "없", "되", "가", "오", "보", "이", "같",
        "그렇", "그러", "주", "말", "나", "들", "놓", "두", "알", "좋", "싶", "안", "뭐", "애", "얘"}

# AI 관찰 규칙의 기준값. calibrate_alerts.py의 모의실험으로 정한다.
ALERT = {"alpha": 0.001, "min_count": 4, "min_books": 3, "min_story_share": 0.6, "window_weeks": 2,
         "baseline_weeks": 4, "min_baseline_weeks": 2, "prior_words": 300, "book_k": 1.0,
         "prior_stories": 6, "story_words": 60}

DISCLAIMER = ("AI는 아이가 한 말의 패턴만 보여 줘요. 진단이 아니니 평소 모습과 함께 판단하고, "
              "걱정되면 원장 선생님이나 전문가와 상의해 주세요.")


def week_start(d):
    d = date.fromisoformat(str(d)[:10])
    return d - timedelta(days=d.weekday())


def load_lexicon():
    lex = pd.read_csv(LEX / "emotion_lexicon.csv")
    lex = lex[lex["감정"].isin(DASHBOARD_EMOTIONS)]
    return {(r.형태소, r.품사): r.감정 for r in lex.itertuples()}


def alert_test(x, n, X, N, pop_rate, n_books, prior_words=ALERT["prior_words"], book_k=ALERT["book_k"]):
    """이번 기간 x회 / n어절이, 지난 기간 X회 / N어절에 비해 유난히 많은지.
    - 아이 기준치가 적을 때 흔들리지 않게 전체 평균(pop_rate, 어절당)을 prior_words어절만큼 섞는다.
    - 그림책마다 분위기가 달라(무서운 책) 횟수가 단순 우연보다 크게 흔들린다. 책 n_books권의
      분위기 차이를 음이항분포(책 하나당 산포 book_k)로 반영해, 무서운 책 한두 권으로는 잘 울리지 않게 한다.
    반환: (p값, 기대 횟수)"""
    rate = (X + prior_words * pop_rate) / (N + prior_words)
    expected = rate * n
    if x <= 0 or expected <= 0:
        return 1.0, expected
    size = book_k * max(n_books, 1)
    return float(nbinom.sf(x - 1, size, size / (size + expected))), expected


def story_presence_test(k, m, K, M, pop_rate, prior_stories=ALERT["prior_stories"],
                        story_words=ALERT["story_words"]):
    """최근 이야기 m편 중 k편에 그 감정이 나왔는지를, 지난 기간 M편 중 K편과 비교 (이항 검정).
    낱말 수가 아니라 '나온 이야기 수'를 세므로, 아주 무서운 이야기 한 편은 1로만 센다.
    지난 기간이 짧을 때를 위해 전체 평균(이야기 한 편에 나올 확률)을 prior_stories편만큼 섞는다."""
    pop_story = 1 - np.exp(-pop_rate * story_words)
    p0 = (K + prior_stories * pop_story) / (M + prior_stories)
    return float(binom.sf(k - 1, m, p0)) if k > 0 else 1.0, p0 * m


class StoryAnalyzer:
    def __init__(self, pop_rates=None, include_psych=False):
        """include_psych=False (기본): 데모 범위인 발달 지표만 낸다.
        True면 감정 흐름(이야기 속 감정 그래프·AI 관찰)과 안전 알림(고백 감지)까지 낸다 (근거 데이터 확보 후 재검토용)."""
        self.include_psych = include_psych
        self.det = Detector()
        self.kiwi = self.det.kiwi
        self.lex = load_lexicon()
        # 어절당 감정 낱말 기준 비율. 실제 Bookids 데이터가 쌓이면 그것으로 바꾼다
        self.pop_rates = pop_rates or {"기쁨": 0.04, "놀람": 0.003, "무서움": 0.004, "슬픔": 0.004, "화남": 0.004}

    def tok(self, text):
        return [(t.form, t.tag.split("-")[0]) for t in self.kiwi.tokenize(text or "")]

    def analyze(self, df):
        """발화마다 어절 수, 감정 낱말, 내용어, 고백 감지 결과를 붙인다."""
        df = df.copy()
        df["week"] = df["date"].map(week_start)
        rows = []
        prev_by_book = {}
        for r in df.sort_values(["child_id", "date", "book_id", "scene"]).itertuples():
            toks = self.tok(r.text)
            q = set(self.tok(getattr(r, "prompt", "") or ""))
            emo, emo_words = Counter(), []
            for j, t in enumerate(toks):
                e = self.lex.get(t)
                if e and t not in q and not is_negated(toks, j) and not is_wish(toks, j):
                    emo[e] += 1
                    emo_words.append(t[0] + ("다" if t[1] in {"VV", "VA"} else ""))
            content = [f + ("다" if g in {"VV", "VA"} else "") for f, g in toks
                       if g in CONTENT_TAGS and f not in STOP and len(f) > 0]
            nouns = [f for f, g in toks if g in {"NNG", "NNP"} and f not in STOP]
            key = (r.child_id, r.book_id)
            ctx = prev_by_book.get(key, [])[-2:]  # 같은 이야기의 앞 장면 2개를 문맥으로
            safety = self.det.detect(r.text, toks=toks, context_toks=ctx) if self.include_psych else []
            prev_by_book.setdefault(key, []).append(toks)
            sents = [s for s in re.split(r"(?<=[.!?])\s+", str(r.text).strip()) if s.strip(" .!?")]
            rows.append({"idx": r.Index, "eojeol": len(str(r.text).split()), "emotions": dict(emo),
                         "emo_words": emo_words, "content": content, "nouns": nouns, "safety": safety,
                         "toks": toks, "lemmas": dev.content_lemmas(toks), "conn": dev.connectives(toks),
                         "sent_eojeol": [len(s.split()) for s in sents] or [0], "ist": dev.ist_terms(toks),
                         "sent_morph": [dev.mlc_m(x) for x in dev.sentences(toks)] or [0],
                         "skills": dev.skills(toks), "quotes": dev.quotes(toks),
                         "pronouns": dev.pronoun_count(toks),
                         "complex": [dev.is_complex(s) for s in dev.sentences(toks)],
                         "complexity": dev.complexity(toks, len(str(r.text).split()))})
        a = pd.DataFrame(rows).set_index("idx")
        return df.join(a)

    # ---- 주간 집계 -------------------------------------------------------------------------

    @staticmethod
    def _week_stats(w):
        scenes = w.groupby(["book_id", "scene"])["used_choice"].any()
        emo = Counter()
        for e in w["emotions"]:
            emo.update(e)
        return {
            "scenes": int(len(scenes)), "self_scenes": int((~scenes).sum()),
            "choice_help": int(w["used_choice"].sum()), "books": int(w["book_id"].nunique()),
            "words": int(w["eojeol"].sum()),
            # 평균 문장 길이: 문장(절) 단위 어절 수 (연구의 MLC-w에 해당)
            "mean_eojeol": float(np.mean([n for ns in w["sent_eojeol"] for n in ns])) if len(w) else None,
            "mlc_m": float(np.mean([n for ns in w["sent_morph"] for n in ns])) if len(w) else None,
            "emotions": emo,
        }

    def child_week_report(self, a, child_id, week, n_trend=6, age_months=None):
        """아이 1명, 1주의 대시보드 수치. a = analyze() 결과, week = 그 주 월요일(date)."""
        c = a[a["child_id"] == child_id]
        this = c[c["week"] == week]
        last = c[c["week"] == week - timedelta(days=7)]
        s, ls = self._week_stats(this), self._week_stats(last)

        # 처음 쓴 낱말: 이 아이의 이전 기록 전체에 없던 내용어
        seen = {w for ws in c[c["week"] < week]["content"] for w in ws}
        new_words = [w for w, _ in Counter(w for ws in this["content"] for w in ws).most_common()
                     if w not in seen]

        # 말하기 변화: 최근 n주 평균 어절
        trend = []
        for k in range(n_trend - 1, -1, -1):
            wk = week - timedelta(days=7 * k)
            ww = c[c["week"] == wk]
            lens = [n for ns in ww["sent_eojeol"] for n in ns]
            trend.append({"week_start": wk.isoformat(),
                          "mean_eojeol": round(float(np.mean(lens)), 2) if lens else None})

        # 자주 나온 인물·주제. 등장인물(토끼·로봇 등)은 이야기마다 바뀌는 게 자연스러워 ↑를 붙이지 않고,
        # 주제어(혼자·밤·엄마 등)만 지난 4주 주평균의 2배 + 1회를 넘으면 ↑
        th = Counter(n for ns in this["nouns"] for n in ns)
        base = c[(c["week"] < week) & (c["week"] >= week - timedelta(days=28))]
        bweeks = max(base["week"].nunique(), 1)
        bth = Counter(n for ns in base["nouns"] for n in ns)
        themes = [{"word": w, "count": n, "character": w in CHARACTER,
                   "increased": w not in CHARACTER and n >= 3 and n >= 2 * bth.get(w, 0) / bweeks + 1}
                  for w, n in th.most_common(8)]

        report = {
            "child_id": child_id,
            "week_start": week.isoformat(), "week_end": (week + timedelta(days=6)).isoformat(),
            "books": s["books"],
            "age_months": age_months,
            "development": self._development(c, week, age_months),
            "cards": {
                "scenes": {"value": s["scenes"], "last_week": ls["scenes"]},
                "self_scenes": {"value": s["self_scenes"], "total": s["scenes"], "choice_help": s["choice_help"]},
                "mlc_m": {"value": round(s["mlc_m"], 1) if s["mlc_m"] else None,
                          "last_week": round(ls["mlc_m"], 1) if ls["mlc_m"] else None},
                "mean_eojeol": {"value": round(s["mean_eojeol"], 1) if s["mean_eojeol"] else None,
                                "last_week": round(ls["mean_eojeol"], 1) if ls["mean_eojeol"] else None},
                "new_words": {"count": len(new_words), "examples": new_words[:3]},
            },
            "speech_trend": trend,
            "themes": themes,
        }
        if self.include_psych:  # 데모 범위 밖: 감정 흐름과 안전 알림
            report["emotions"] = [{"name": e, "this_week": s["emotions"].get(e, 0),
                                   "last_week": ls["emotions"].get(e, 0),
                                   "increased": s["emotions"].get(e, 0) > ls["emotions"].get(e, 0)}
                                  for e in DASHBOARD_EMOTIONS]
            report["observation"] = self._observation(c, week)
            report["safety"] = [{"level": x["level"], "category": x["category"], "reason": x["reason"],
                                 "text": r.text, "date": r.date, "book_id": r.book_id, "scene": int(r.scene)}
                                for r in this.itertuples() for x in r.safety]
        return report

    @staticmethod
    def _books(rows):
        """발화 행들 -> 이야기 목록 [{book_id, utts, elements}] (장면 순서대로)."""
        out = []
        for bid, b in rows.sort_values(["date", "scene"]).groupby("book_id", sort=False):
            utts = [{"eojeol": r.eojeol, "sent_eojeol": r.sent_eojeol, "sent_morph": r.sent_morph, "lemmas": r.lemmas, "conn": r.conn,
                     "ist": r.ist, "used_choice": bool(r.used_choice)} for r in b.itertuples()]
            g = dev.story_grammar(list(b["toks"]))
            out.append({"book_id": bid, "utts": utts, "grammar": g, "level": dev.episode_level(g),
                        "intro": dev.intro_explicit(list(b["toks"])),
                        "quotes": int(b["quotes"].sum()), "pronouns": int(b["pronouns"].sum()),
                        "complex": [x for xs in b["complex"] for x in xs]})
        return out

    @staticmethod
    def _refined(books):
        """세부 지표 (판정 없이 값만): 에피소드 단계 분포, 인물 소개, 인용, 복문, 지칭어."""
        if not books:
            return None
        n_sent = sum(len(b["complex"]) for b in books) or 1
        levels = Counter(b["level"] for b in books)
        return {
            "stories": len(books),
            "ndw50": None if dev.ndw50(books) is None else round(dev.ndw50(books), 1),
            "episode_levels": {name: levels.get(i, 0) for i, name in enumerate(dev.EPISODE_LEVELS)},
            "best_level": dev.EPISODE_LEVELS[max(levels)],
            "intro_explicit_pct": round(100 * float(np.mean([b["intro"] for b in books]))),
            "quotes_per_story": round(float(np.mean([b["quotes"] for b in books])), 2),
            "complex_pct": round(100 * sum(sum(b["complex"]) for b in books) / n_sent),
            "pronouns_per_10_sent": round(10 * sum(b["pronouns"] for b in books) / n_sent, 2),
        }

    @staticmethod
    def _evidence(rows):
        """그 기간의 '중간 수준' 발화 (가장 잘한 문장이 아니라 보통 문장을 보여 줘서 과장하지 않는다)."""
        if rows.empty:
            return None
        med = rows["complexity"].median()
        r = rows.iloc[(rows["complexity"] - med).abs().argsort().iloc[0]]
        return {"text": r.text, "date": r.date, "complexity": round(float(r.complexity), 1)}

    def _development(self, c, week, age_months=None):
        """발달 지표: 이번 주 vs 지난 4주, 그리고 6주 성장 (앞 3주 vs 뒤 3주)."""
        this = c[c["week"] == week]
        base = c[(c["week"] < week) & (c["week"] >= week - timedelta(days=28))]
        recent = c[(c["week"] > week - timedelta(days=21)) & (c["week"] <= week)]
        early = c[(c["week"] > week - timedelta(days=42)) & (c["week"] <= week - timedelta(days=21))]
        bt, bb = self._books(this), self._books(base)

        # 이야기문법: 단위별로 그 단위가 들어간 이야기 비율, 평균 점수, 완전한 에피소드 수 (이번 주)
        grammar = {
            "units": {u: (round(100 * float(np.mean([b["grammar"]["units"][u] for b in bt]))) if bt else None)
                      for u in dev.SG_UNITS},
            "mean_score": round(float(np.mean([b["grammar"]["score"] for b in bt])), 1) if bt else None,
            "complete_episodes": int(sum(b["grammar"]["complete_episodes"] for b in bt)),
            "stories": len(bt),
        }
        # 마음을 나타내는 말: 종류별 횟수와 예
        ist = Counter(k for ts in this["ist"] for k, _ in ts)
        ist_words = {k: sorted({w for ts in this["ist"] for kk, w in ts if kk == k})[:3] for k in dev.IST_TYPES}
        # 이음말 종류와 예
        conn = Counter(k for ks in this["conn"] for k in ks)
        conn_ex = next((r.text for r in this.itertuples() if r.conn), None)
        # 감정 어휘의 폭: 이번 주에 쓴 서로 다른 감정 낱말, 지금까지 누적, 이번 주 처음 쓴 것
        before = {w for ws in c[c["week"] < week]["emo_words"] for w in ws}
        now = {w for ws in this["emo_words"] for w in ws}

        # 처음 해낸 것: 그 전 3주에는 한 번도 없었고, 최근 3주에 서로 다른 이야기 2편 이상에서 나온 기술.
        # 한 번만 나온 것은 우연일 수 있어서 '자리 잡은' 경우만 보여 준다 (그 전 기록이 2주 이상 있을 때만)
        milestones = []
        hist = c[c["week"] <= week].sort_values(["date", "book_id", "scene"])
        start = week - timedelta(days=14)  # 최근 3주의 첫 주
        if early["week"].nunique() >= 2:
            for skill in dev.SKILLS:
                hit = hist[hist["skills"].map(lambda s: skill in s)]
                if len(hit) and hit["week"].min() >= start and hit["book_id"].nunique() >= 2:
                    r = hit.iloc[0]
                    milestones.append({"skill": skill, "date": r.date, "text": r.text,
                                       "stories": int(hit["book_id"].nunique())})
            hist_books = self._books(hist)
            done = [b for b in hist_books if b["grammar"]["complete_episodes"]]
            first_week = lambda b: hist[hist.book_id == b["book_id"]]["week"].iloc[0]
            if len(done) >= 2 and first_week(done[0]) >= start:
                rows = hist[hist.book_id == done[0]["book_id"]]
                milestones.append({"skill": "완전한 에피소드 (목표→시도→결과)", "date": rows["date"].iloc[0],
                                   "text": " ".join(rows["text"].tolist()[:4]) + " …", "stories": len(done)})

        # 다음 단계 질문: 최근 3주 이야기 기준, 아이가 최근 가장 많이 쓴 등장인물 이름으로
        recent_books = self._books(recent)
        recent_skills = {s for ss in recent["skills"] for s in ss}
        chars = Counter(n for ns in recent["nouns"] for n in ns if n in CHARACTER)
        character = chars.most_common(1)[0][0] if chars else "주인공"

        # 그래프용: 지표별 최근 6주 주간 값, 최근 3주 이야기문법 단위 비율
        weekly = {}
        for k in dev.METRICS:
            f = dev.METRICS[k][2]
            weekly[k] = []
            for i in range(5, -1, -1):
                bks = self._books(c[c["week"] == week - timedelta(days=7 * i)])
                v = f(bks) if bks else None
                weekly[k].append(None if v is None else round(v, 2))
        units_3w = {u: (round(100 * float(np.mean([b["grammar"]["units"][u] for b in recent_books])))
                        if recent_books else None) for u in dev.SG_UNITS}
        ist_3w = Counter(k for ts in recent["ist"] for k, _ in ts)
        conn_3w = Counter(k for ks in recent["conn"] for k in ks)

        return {
            "weekly": weekly,
            "recent_3w": {"units": units_3w, "ist": {k: ist_3w.get(k, 0) for k in dev.IST_TYPES},
                          "conn": {k: conn_3w.get(k, 0) for k in ["원인", "순서", "대조"]},
                          "sentences": int(sum(len(x) for x in recent["sent_eojeol"])),
                          "sentences_per_story": round(float(np.mean([len(b["utts"]) and sum(len(u["sent_eojeol"]) for u in b["utts"]) for b in recent_books])), 1) if recent_books else None},
            "age_ref": ({**dev.AGE_REF[dev.age_group(age_months)], "mlc_m": dev.MLC_M_REF[dev.age_group(age_months)],
                         "age": dev.age_group(age_months)} if dev.age_group(age_months) in dev.AGE_REF else None),
            "growth_evidence": {"before": self._evidence(early), "recent": self._evidence(recent),
                                "note": "각 기간의 중간 수준 장면 (가장 잘한 장면이 아님)"},
            "milestones": milestones,
            "next_steps": dev.next_steps(recent_books, recent_skills, character, age_months),
            "refined": {"recent_3w": self._refined(recent_books), "before_3w": self._refined(self._books(early))},
            "this_week_vs_4weeks": dev.compare(bt, bb, test=False),   # 값만 (판정 없음)
            "six_week_growth": dev.compare(self._books(recent), self._books(early)),  # 최근 3주 vs 그 전 3주
            "story_grammar": grammar,
            "internal_states": {"by_type": {k: ist.get(k, 0) for k in dev.IST_TYPES}, "examples": ist_words},
            "connectives": {"by_type": dict(conn), "example": conn_ex},
            "emotion_vocab": {"this_week": sorted(now), "cumulative": len(before | now),
                              "new": sorted(now - before)},
        }

    def _observation(self, c, week):
        """그 아이의 지난 4주에 비해 최근 2주에 유난히 늘어난 부정 감정이 있으면 AI 관찰 메모."""
        cfg = ALERT
        start = week - timedelta(days=7 * (cfg["window_weeks"] - 1))
        base = c[(c["week"] < start) & (c["week"] >= start - timedelta(days=7 * cfg["baseline_weeks"]))]
        this = c[(c["week"] >= start) & (c["week"] <= week)]
        if base["week"].nunique() < cfg["min_baseline_weeks"] or this.empty:
            return None
        m, M = this["book_id"].nunique(), base["book_id"].nunique()
        found = []
        for e in WATCH_EMOTIONS:
            has = lambda rows: rows[rows["emotions"].map(lambda d: d.get(e, 0) > 0)]
            hit = has(this)
            k, K = hit["book_id"].nunique(), has(base)["book_id"].nunique()
            x = sum(d.get(e, 0) for d in this["emotions"])
            X = sum(d.get(e, 0) for d in base["emotions"])
            # 세기(횟수 검정)와 퍼짐(최근 이야기의 60% 이상, 전보다 높은 비율)을 둘 다 만족해야 한다.
            # 무서운 이야기 한 편은 퍼짐에서, 원래 가끔 무서움을 말하던 아이는 세기에서 걸러진다.
            p, _ = alert_test(x, this["eojeol"].sum(), X, base["eojeol"].sum(), self.pop_rates[e], m)
            spread = k >= cfg["min_books"] and k / m >= cfg["min_story_share"] and k / m > K / max(M, 1)
            if x >= cfg["min_count"] and spread and p < cfg["alpha"]:
                found.append((p, e, x, X, k, m, K, M, base["week"].nunique(), hit))
        if not found:
            return None
        p, e, x, X, k, m, K, M, nweeks, hit = min(found)
        # 등장인물(펭귄·토끼 등)은 이야기마다 바뀌므로 핵심 낱말에서 뺀다
        words = [w for w, _ in Counter(w for t in hit["content"] for w in t if w not in CHARACTER).most_common(2)]
        ev = hit.sort_values("date").tail(2)
        return {
            "emotion": e,
            "title": f"최근 이야기에 {e} 표현이 자주 나왔어요",
            "detail": f"최근 {cfg['window_weeks']}주 이야기 {m}편 중 {k}편에 나왔어요 ({x}번). "
                      f"그 전 {nweeks}주에는 {M}편 중 {K}편이었어요.",
            "this_week": int(x), "baseline": int(X), "stories": {"recent": [int(k), int(m)], "before": [int(K), int(M)]},
            "baseline_weeks": int(nweeks), "p_value": float(f"{p:.2g}"),
            "top_words": words,
            "evidence": [{"text": r.text, "date": r.date, "book_id": r.book_id, "scene": int(r.scene)}
                         for r in ev.itertuples()],
            "suggestion": "자유놀이 시간에 이 그림책을 함께 보며 이야기를 나눠 보세요.",
            "disclaimer": DISCLAIMER,
        }
