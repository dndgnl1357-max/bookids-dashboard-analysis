"""2단계 EDA: 라벨 분포, 불균형, 아이 대답 길이.

사용: .venv/Scripts/python scripts/eda.py
출력: reports/eda.md, reports/fig_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
REPORTS = ROOT / "reports"

plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기본 한글 글꼴
plt.rcParams["axes.unicode_minus"] = False

def main():
    REPORTS.mkdir(exist_ok=True)
    c = pd.read_parquet(TABLES / "children.parquet")
    u = pd.read_parquet(TABLES / "utterances.parquet")
    out = []

    def section(title, obj):
        out.append(f"\n## {title}\n")
        out.append("```\n" + (obj.to_string() if hasattr(obj, "to_string") else str(obj)) + "\n```")

    # 1. 아동 기본 분포
    section("split별 인원", c.split.value_counts())
    for col in ["나이", "성별", "학년", "유형구분", "가정환경", "위기단계", "학대의심"]:
        if col in c.columns:
            section(f"{col} 분포 (train / val)",
                    pd.crosstab(c[col], c.split, margins=True))

    # 2. 위기단계와 다른 변수의 관계
    section("위기단계 x 유형구분 (행 비율 %)",
            (pd.crosstab(c["유형구분"], c["위기단계"], normalize="index") * 100).round(1))
    section("위기단계 x 나이 (행 비율 %)",
            (pd.crosstab(c["나이"], c["위기단계"], normalize="index") * 100).round(1))
    section("위기단계별 합계점수", c.groupby("위기단계")["합계점수"].describe().round(1))

    # 3. 영역별 위기단계
    dom_cols = [col for col in c.columns if col.endswith("_위기단계")]
    dom = pd.concat({col.replace("_위기단계", ""): c[col].value_counts() for col in dom_cols}, axis=1)
    section("영역별 위기단계 인원", dom.fillna(0).astype(int))

    # 4. 아이 대답(A) 길이
    a = u[u.type == "A"].copy()
    a["text"] = a["text"].fillna("").str.strip()
    a["chars"] = a["text"].str.len()
    a["eojeol"] = a["text"].str.split().str.len()
    a = a.merge(c[["split", "ID", "나이", "위기단계"]], on=["split", "ID"])

    section("아이 대답 길이 (어절)", a["eojeol"].describe().round(2))
    section("나이별 평균 어절 수", a.groupby("나이")["eojeol"].agg(["mean", "median", "count"]).round(2))
    section("위기단계별 평균 어절 수", a.groupby("위기단계")["eojeol"].agg(["mean", "median"]).round(2))
    short = a["eojeol"] <= 1
    section("한 어절 이하 대답 비율 %", round(short.mean() * 100, 1))
    section("가장 흔한 대답 상위 20", a["text"].value_counts().head(20))

    per_child = a.groupby(["split", "ID"]).agg(answers=("text", "size"), words=("eojeol", "sum"))
    section("아동 1명당 대답 수 / 총 어절", per_child.describe().round(1))

    # 그림
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c["위기단계"].value_counts().plot.bar(ax=ax[0], color="#777")
    ax[0].set_title("위기단계 인원")
    a.groupby("나이")["eojeol"].mean().plot.bar(ax=ax[1], color="#777")
    ax[1].set_title("나이별 아이 대답 평균 어절 수")
    for x in ax:
        x.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_overview.png", dpi=120)

    (REPORTS / "eda.md").write_text("# EDA 결과\n" + "\n".join(out), encoding="utf-8")
    print("saved reports/eda.md, reports/fig_overview.png")


if __name__ == "__main__":
    main()
