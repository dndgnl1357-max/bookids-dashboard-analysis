"""음성 실험 0단계: 라벨의 시간 정보만으로 만드는 말하기 타이밍 특징 (음성 파일 불필요).

사용: .venv/Scripts/python scripts/timing_features.py
출력: data/tables/answer_timing.parquet (대답별), data/tables/child_timing.parquet (아동별), reports/timing_check.md

- 반응 지연: 상담사 질문이 끝난 뒤 아이 대답이 시작되기까지 (초)
- 대답 길이: 대답 시작 ~ 끝 (초)
- 말 속도: 한글 음절 수 / 대답 길이 (음절/초)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, REPORTS = ROOT / "data" / "tables", ROOT / "reports"
HANGUL = re.compile(r"[가-힣]")


def main():
    u = pd.read_parquet(TABLES / "utterances.parquet")
    u = u.sort_values(["split", "ID", "start"]).reset_index(drop=True)
    # 바로 앞 발화가 질문(Q)인 대답(A)만: 반응 지연 = 대답 시작 - 질문 끝
    prev = u.groupby(["split", "ID"]).shift(1)
    a = u[(u["type"] == "A") & (prev["type"] == "Q")].copy()
    a["latency"] = a["start"] - prev.loc[a.index, "end"]
    a["syll"] = a["text"].fillna("").map(lambda s: len(HANGUL.findall(s)))
    a["rate"] = a["syll"] / a["dur"].where(a["dur"] > 0)

    # 시간 정보 점검: 음수 지연, 0 이하 길이, 비현실적인 말 속도
    checks = {
        "대답 수": len(a),
        "지연 < 0 (질문과 겹침)": int((a["latency"] < 0).sum()),
        "지연 > 30초": int((a["latency"] > 30).sum()),
        "길이 <= 0": int((a["dur"] <= 0).sum()),
        "말 속도 > 12음절/초 (비현실적)": int((a["rate"] > 12).sum()),
        "말 속도 < 0.5음절/초": int((a["rate"] < 0.5).sum()),
    }
    ok = a["latency"].between(0, 30) & (a["dur"] > 0) & a["rate"].between(0.5, 12)
    a["valid"] = ok
    a[["split", "ID", "영역", "항목", "turn", "text", "start", "end", "dur", "latency", "syll", "rate", "valid"]] \
        .to_parquet(TABLES / "answer_timing.parquet", index=False)

    v = a[ok]
    g = v.groupby(["split", "ID"])
    child = pd.DataFrame({
        "latency_med": g["latency"].median(),
        "latency_iqr": g["latency"].quantile(0.75) - g["latency"].quantile(0.25),
        "long_latency_ratio": g["latency"].agg(lambda s: (s > 3).mean()),
        "dur_med": g["dur"].median(),
        "rate_med": g["rate"].median(),
        "n_valid": g.size(),
    }).reset_index()
    child.to_parquet(TABLES / "child_timing.parquet", index=False)

    c = pd.read_parquet(TABLES / "children.parquet", columns=["split", "ID", "위기단계", "나이"])
    m = child.merge(c, on=["split", "ID"])
    desc = v[["latency", "dur", "rate"]].describe(percentiles=[.05, .25, .5, .75, .95]).round(2)
    by_level = m.groupby("위기단계")[["latency_med", "long_latency_ratio", "dur_med", "rate_med"]].mean() \
        .reindex(["정상군", "관찰필요", "상담필요", "학대의심", "응급"]).round(3)
    lines = ["# 음성 실험 0단계: 타이밍 특징 점검", "",
             "## 시간 정보 이상치", "```\n" + "\n".join(f"{k}: {v:,}" for k, v in checks.items()) + "\n```", "",
             f"## 유효한 대답 {ok.sum():,}개의 분포 (초, 음절/초)", "```\n" + desc.to_string() + "\n```", "",
             "## 위기단계별 아동 평균", "```\n" + by_level.to_string() + "\n```"]
    (REPORTS / "timing_check.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved reports/timing_check.md")


if __name__ == "__main__":
    main()
