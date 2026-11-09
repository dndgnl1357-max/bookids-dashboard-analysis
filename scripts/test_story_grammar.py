"""이야기문법 자동 채점 점검: 직접 만든 이야기들이 기대한 단위·점수 순서로 나오는지.

사용: .venv/Scripts/python scripts/test_story_grammar.py
"""
from kiwipiepy import Kiwi

import dev_metrics as dev

# (이름, 장면 목록, 기대: 들어가야 할 단위, 빠져야 할 단위, 완전한 에피소드 수)
CASES = [
    ("나열만 한 이야기", ["토끼가 갔어요.", "놀았어요.", "먹었어요.", "잤어요."],
     set(), {"목표", "결과"}, 0),
    ("배경만 있는 이야기", ["어느 날 토끼가 숲에 갔어요.", "토끼가 노래를 불렀어요.", "집에 갔어요."],
     {"배경"}, {"목표"}, 0),
    ("문제-해결 (목표 없음)", ["옛날에 숲에 곰이 살았어요.", "그런데 곰이 공을 잃어버렸어요.",
                         "곰이 여기저기 다녔어요.", "드디어 공을 찾았어요.", "곰은 너무 기뻤어요."],
     {"배경", "시작 사건", "시도", "결과", "반응"}, {"목표"}, 0),
    ("완전한 에피소드", ["어느 날 토끼가 바다에 갔어요.", "갑자기 토끼가 모자를 잃어버렸어요.",
                   "토끼는 모자를 찾고 싶었어요.", "토끼가 열심히 찾아봤어요.", "드디어 모자를 찾았어요.",
                   "토끼는 정말 행복했어요."],
     {"배경", "시작 사건", "목표", "시도", "결과", "반응"}, set(), 1),
    ("에피소드 두 개", ["옛날에 숲에 여우가 살았어요.", "여우는 배가 고팠어요.", "여우는 포도를 먹으려고 했어요.",
                   "여우가 나무에 올라가 봤어요.", "하지만 못 먹었어요.", "여우는 속상했어요.",
                   "그다음에 여우는 친구를 만나고 싶었어요.", "여우가 친구 집에 달려갔어요.", "결국 친구를 만났어요.",
                   "여우는 신났어요."],
     {"배경", "시작 사건", "목표", "시도", "결과", "반응"}, set(), 2),
]


def main():
    kiwi = Kiwi()
    tok = lambda s: [(t.form, t.tag.split("-")[0]) for t in kiwi.tokenize(s)]
    scores, fails = [], 0
    for name, scenes, must, must_not, complete in CASES:
        g = dev.story_grammar([tok(s) for s in scenes])
        have = {u for u, v in g["units"].items() if v}
        ok = must <= have and not (must_not & have) and g["complete_episodes"] == complete
        fails += not ok
        scores.append(g["score"])
        print(f"{'OK ' if ok else 'FAIL'} {name}: 점수 {g['score']}/17, 완전한 에피소드 {g['complete_episodes']}, "
              f"단위 {sorted(have)}" + ("" if ok else f"  (기대: 포함 {sorted(must)}, 제외 {sorted(must_not)}, 완전 {complete})"))
    order_ok = scores == sorted(scores)
    print(f"점수 순서 (단순 → 복잡) {'OK' if order_ok else 'FAIL'}: {scores}")
    ist = dev.ist_terms(tok("토끼가 꽃을 보고 생각했어요. 배가 고파서 엄마를 불렀어요. 말이 달렸어요."))
    print("마음을 나타내는 말 예:", ist)
    print(f"{len(CASES) - fails}/{len(CASES)} 통과")

    # 세부 지표: 인용 · 복문 · 지칭어
    refined = [("토끼가 \"같이 놀자\"라고 말했어요.", "인용", 1), ("토끼가 같이 놀자고 했어요.", "인용", 1),
               ("토끼가 놀았어요.", "인용", 0), ("배가 고픈 토끼가 사과를 먹었어요.", "복문", True),
               ("작은 토끼가 살았어요.", "복문", False), ("토끼가 배가 고파서 울었어요.", "복문", True),
               ("토끼가 놀았어요.", "복문", False), ("걔가 그거 먹었어요.", "지칭어", 2), ("토끼가 사과를 먹었어요.", "지칭어", 0)]
    ok = 0
    for s, kind, exp in refined:
        t = tok(s)
        got = {"인용": dev.quotes(t), "복문": dev.is_complex(dev.sentences(t)[0]), "지칭어": dev.pronoun_count(t)}[kind]
        ok += got == exp
        if got != exp:
            print(f"FAIL {kind}: {s} 기대 {exp} 결과 {got}")
    print(f"세부 지표 {ok}/{len(refined)} 통과")


if __name__ == "__main__":
    main()
