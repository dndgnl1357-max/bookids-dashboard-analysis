"""그림책 이야기 같은 문장을 넣어 모델이 오경보를 내는지 확인한다.

사용: .venv/Scripts/python scripts/probe_story.py  (model.py 실행 후)
상담 문답으로 학습한 모델이 '이야기 속 사건'과 '아이 본인의 고백'을 구분하는지 보는 점검용.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from kiwipiepy import Kiwi

from model import split_tokens  # noqa: F401  joblib이 토크나이저 함수를 찾을 수 있게

ROOT = Path(__file__).resolve().parents[1]

STORIES = {
    "평범한 이야기": ["토끼가 숲에 놀러 갔어요.", "다람쥐랑 같이 도토리를 먹었어요.",
                "재밌게 놀고 집에 갔어요.", "엄마가 안아 줬어요."],
    "이야기 속 갈등 (흔한 줄거리)": ["늑대가 나타났어요.", "늑대가 토끼를 때렸어요.", "토끼가 무서워서 울었어요.",
                          "곰이 와서 늑대를 혼냈어요.", "늑대는 도망갔어요."],
    "공룡 싸움 이야기": ["공룡 두 마리가 싸웠어요.", "티라노가 물어서 공룡이 죽었어요.",
                  "다른 공룡이 화났어요.", "그래서 또 싸웠어요."],
    "아이 본인 고백처럼 들리는 말": ["아빠가 술 마시고 나를 때렸어요.", "너무 아팠어요.", "무서워서 숨었어요."],
}


def main():
    kiwi = Kiwi()
    enc = lambda s: " ".join(f"{t.form}/{t.tag.split('-')[0]}" for t in kiwi.tokenize(s))
    res = json.loads((ROOT / "reports" / "model_results.json").read_text(encoding="utf-8"))
    # F+T는 아동 단위 특징(100어절당 등)이 필요해 문장 몇 개로는 점검할 수 없어 T만 본다
    for name in ["T"]:
        model = joblib.load(ROOT / "models" / f"{name}_binary.joblib")
        cut = res["models"][name]["binary"]["threshold"]
        print(f"== {name} 모델 (기준점 {cut:.2f})")
        for story, sents in STORIES.items():
            doc = " ".join(enc(s) for s in sents)
            p = model.predict_proba(pd.DataFrame({"doc_a": [doc]}))[0, 1]
            print(f"  {story:28s} 위험확률 {p:.2f}  {'-> 위험 알림' if p >= cut else ''}")


if __name__ == "__main__":
    main()
