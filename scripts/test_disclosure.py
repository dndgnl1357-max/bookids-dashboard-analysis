"""고백 감지 점검: tests/disclosure_cases.csv의 문장이 기대대로 판정되는지.

사용: .venv/Scripts/python scripts/test_disclosure.py
문장은 그림책 이야기와 고백을 구분하는지 보려고 직접 만든 것이다 (실제 아이 발화 아님).
"""
from pathlib import Path

import pandas as pd

from disclosure import Detector, summarize

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = pd.read_csv(ROOT / "tests" / "disclosure_cases.csv").fillna("")
    det = Detector()
    fails = 0
    for r in cases.itertuples():
        found = det.detect(r[3], context=[r[2]] if r[2] else [])
        got = summarize(found) or "없음"
        ok = got == r.기대
        fails += not ok
        why = f"  ({found[0]['reason']}, {found[0]['agent']}→{found[0]['victim']})" if found else ""
        print(f"{'OK ' if ok else 'FAIL'} [{r.구분}] {r[3]}  기대={r.기대} 결과={got}{why}")
    print(f"\n{len(cases) - fails}/{len(cases)} 통과")


if __name__ == "__main__":
    main()
