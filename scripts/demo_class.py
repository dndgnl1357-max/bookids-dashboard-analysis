"""가상의 반(18명 × 6주) 이야기 기록을 만들어 발달 지표를 처음부터 끝까지 돌려 본다.

사용: .venv/Scripts/python scripts/demo_class.py
출력:
  data/demo/story_log.csv          가상 발화 기록 (실제 아이 말이 아님)
  outputs/dashboard_week.json      웹 담당에게 넘길 대시보드 데이터 예시 (마지막 주)

확인하려는 상황
  아이 D: 6주 동안 뚜렷하게 성장 -> 6주 성장 '늘었어요', 처음 해낸 것, 근거 문장이 달라짐
  아이 G: 6주 동안 뚜렷하게 줄어듦 -> 줄어든 지표가 보이거나, 적어도 '늘었어요'가 붙지 않음
  나머지 16명: 수준이 일정 -> '늘었어요/줄었어요'가 붙지 않아야 함
"""
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from story_metrics import StoryAnalyzer

ROOT = Path(__file__).resolve().parents[1]
DEMO, OUT = ROOT / "data" / "demo", ROOT / "outputs"
RNG = np.random.default_rng(7)
LAST_WEEK = date(2026, 11, 2)
WEEKS = [LAST_WEEK - timedelta(days=7 * k) for k in range(5, -1, -1)]
CHILDREN = [f"아이 {chr(ord('A') + i)}" for i in range(18)]

CHARS = ["토끼", "다람쥐", "곰", "여우", "강아지", "고양이", "공룡", "공주", "로봇", "펭귄"]
PLACES = ["숲", "바다", "공원", "하늘", "동굴", "놀이터", "기차역"]

# 수준이 낮을수록: 짧은 말, 지칭어(걔·그거), 나열
SHORT = ["{c}{ga} 갔어요.", "놀았어요.", "먹었어요.", "{c}{ga} 잤어요.", "집에 갔어요.", "{c}{ga} 웃었어요.",
         "친구 만났어요.", "걔가 갔어요.", "그거 먹었어요.", "걔가 울었어요.", "끝."]
ACTIONS = ["{c}{ga} {p}에 갔어요.", "{c}{ga} 친구를 만났어요.", "{c}{neun} 도토리를 찾았어요.",
           "{c}{ga} 맛있는 케이크를 먹었어요.", "{c}{ga} 노래를 불렀어요.", "같이 {p}에서 놀았어요.",
           "{c}{ga} 집으로 돌아갔어요.", "{c}{ga} 큰 나무에 올라갔어요.", "{c}{neun} 그림을 그렸어요.",
           "{c}{neun} 너무 행복했어요.", "{c}{ga} 깜짝 놀랐어요.", "정말 재밌었어요."]
# 수준이 높을수록: 다양한 낱말, 이음말, 꾸미는 절, 인용, 생각을 나타내는 말
RICH = ["{c}{ga} 반짝이는 조개껍데기를 주웠어요.", "{c}{neun} 커다란 무지개를 바라봤어요.",
        "{c}{ga} 친구에게 편지를 썼어요.", "{c}{ga} 자전거를 타고 달렸어요.", "{c}{neun} 따뜻한 수프를 끓였어요.",
        "{c}{ga} 밤하늘의 별을 세었어요.", "{c}{neun} 빨간 풍선을 날렸어요.", "{c}{ga} 모래성을 쌓았어요.",
        "{c}{neun} 신비한 지도를 펼쳤어요.", "{c}{ga} 무거운 상자를 옮겼어요."]
CONNECTED = ["{c}{ga} 배가 고파서 사과를 먹었어요.", "그래서 {c}{neun} 집으로 갔어요.", "그런데 갑자기 비가 왔어요.",
             "{c}{ga} 놀다가 넘어졌어요.", "{c}{ga} 졸려서 낮잠을 잤어요.", "하지만 {c}{neun} 포기하지 않았어요.",
             "친구가 오니까 {c}{neun} 신났어요.", "{c}{ga} \"같이 놀자\"라고 말했어요.",
             "배가 고픈 {c}{ga} 사과를 찾았어요.", "{c}{neun} 친구가 준 선물을 열어 봤어요.",
             "{c}{neun} 집에 가야겠다고 생각했어요.", "{c}{neun} 친구가 어디 있는지 궁금했어요."]
ARC = {"setting": ["어느 날 {c}{ga} {p}에 갔어요.", "옛날에 {p}에 {c}{ga} 살았어요."],
       "problem": ["그런데 {c}{ga} 길을 잃어버렸어요.", "{c}{neun} 소중한 공을 잃어버렸어요.",
                   "{c}{ga} 넘어져서 다리가 아팠어요."],
       "goal": ["{c}{neun} 공을 찾고 싶었어요.", "{c}{ga} 집에 돌아가려고 했어요.", "{c}{neun} 친구를 만나고 싶었어요."],
       "attempt": ["{c}{ga} 여기저기 열심히 찾아봤어요.", "{c}{ga} 큰 소리로 친구를 불렀어요.", "{c}{ga} 나무에 올라가 봤어요."],
       "resolve": ["친구가 와서 {c}{reul} 도와줬어요.", "{c}{ga} 드디어 공을 찾았어요.", "결국 {c}{ga} 집을 찾았어요."],
       "ending": ["그래서 {c}{neun} 행복하게 집으로 돌아갔어요. 끝.", "{c}{neun} 친구랑 같이 잠을 잤어요. 끝."]}
LEVEL = {"아이 D": lambda wi: 0.05 + 0.18 * wi,   # 6주 동안 뚜렷하게 성장 (시연용)
         "아이 G": lambda wi: 0.90 - 0.16 * wi}   # 6주 동안 뚜렷하게 줄어듦 (시연용)


def josa(word, pair):
    has_batchim = (ord(word[-1]) - 0xAC00) % 28 != 0
    return pair[0] if has_batchim else pair[1]


def fill(t, c):
    return t.format(c=c, p=RNG.choice(PLACES), ga=josa(c, ("이", "가")), neun=josa(c, ("은", "는")),
                    reul=josa(c, ("을", "를")))


def scene_text(level, c, role):
    if role:
        return fill(RNG.choice(ARC[role]), c)
    r = RNG.random()
    if r < 0.6 * level:
        return fill(RNG.choice(CONNECTED), c)
    if r < 1.1 * level:
        return fill(RNG.choice(RICH), c)
    if RNG.random() < 1 - level:
        return fill(RNG.choice(SHORT), c)
    return fill(RNG.choice(ACTIONS), c)


def make_log():
    rows = []
    base_level = {ch: RNG.uniform(0.3, 0.6) for ch in CHILDREN}
    for child in CHILDREN:
        for wi, wk in enumerate(WEEKS):
            level = LEVEL[child](wi) if child in LEVEL else base_level[child]
            for b in range(RNG.integers(2, 4)):
                book = f"{child}-{wk:%m%d}-{b + 1}"
                c = RNG.choice(CHARS)
                day = wk + timedelta(days=int(RNG.integers(0, 5)))
                n_scene = int(RNG.integers(6, 10))
                roles = {}
                if RNG.random() < level:  # 배경-문제-해결-끝이 있는 이야기
                    roles = {0: "setting", 1: "problem", n_scene - 2: "resolve", n_scene - 1: "ending"}
                    if RNG.random() < level:  # 수준이 높으면 목표와 시도까지 (완전한 에피소드)
                        roles.update({2: "goal", 3: "attempt"})
                for s in range(n_scene):
                    rows.append({"child_id": child, "date": (day).isoformat(), "book_id": book, "scene": s + 1,
                                 "text": scene_text(level, c, roles.get(s)), "prompt": "그다음엔 어떻게 될까?",
                                 "used_choice": bool(RNG.random() < 0.35 - 0.25 * level)})
    return pd.DataFrame(rows)


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    log = make_log()
    log.to_csv(DEMO / "story_log.csv", index=False, encoding="utf-8-sig")
    print(f"log: {len(log)} utterances")

    # 나이(개월): 4세 반과 5세가 섞인 반. 발화 생성과 따로 뽑아 기존 가상 발화는 그대로 둔다
    ages = {c: int(m) for c, m in zip(CHILDREN, np.random.default_rng(11).integers(50, 72, len(CHILDREN)))}
    an = StoryAnalyzer()  # 발달 지표만 (감정 흐름·안전 알림 제외)
    a = an.analyze(log)
    reports = {c: an.child_week_report(a, c, LAST_WEEK, age_months=ages[c]) for c in CHILDREN}
    growth = lambda r, d: [m["label"] for m in r["development"]["six_week_growth"] if m["change"] == d]
    out = {
        "class": {"name": "햇살반", "n_children": len(CHILDREN), "week_start": LAST_WEEK.isoformat(),
                  "consent": {"given": len(CHILDREN), "total": len(CHILDREN)}, "synthetic": True,
                  "scope": "발달 지표 데모 (감정 흐름·안전 알림 제외)"},
        "children": [{"child_id": c, "age_months": ages[c], "growth_up": growth(reports[c], "up"), "growth_down": growth(reports[c], "down"),
                      "milestones": len(reports[c]["development"]["milestones"])} for c in CHILDREN],
        "reports": reports,
    }
    (OUT / "dashboard_week.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    for k in out["children"]:
        flags = []
        if k["growth_up"]:
            flags.append("늘었어요: " + ", ".join(k["growth_up"]))
        if k["growth_down"]:
            flags.append("줄었어요: " + ", ".join(k["growth_down"]))
        if k["milestones"]:
            flags.append(f"처음 해낸 것 {k['milestones']}개")
        print(f"{k['child_id']}: {'; '.join(flags) or '-'}")
    print("saved outputs/dashboard_week.json")


if __name__ == "__main__":
    main()
