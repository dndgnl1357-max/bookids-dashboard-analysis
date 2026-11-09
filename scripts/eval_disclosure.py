"""고백 감지를 실제 상담 데이터에 돌려 본다.

사용: .venv/Scripts/python scripts/eval_disclosure.py  (features.py 실행 후, 형태소 캐시 사용)
출력:
  data/tables/disclosure_hits.parquet  감지된 문장
  reports/disclosure_eval.md           아동 단위 감지율 (위기단계별, 학대 유형별)
  reports/disclosure_samples.md        사람이 검토할 표본 문장

문장 단위 정답 라벨이 없으므로, 아동 단위로 본다:
학대의심 아동에게서 많이, 정상군 아동에게서 적게 감지돼야 한다.
"""
from pathlib import Path

import pandas as pd

from disclosure import Detector, summarize

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
REPORTS = ROOT / "reports"

def dec(s):
    """캐시 문자열 -> [(형태소, 품사)]. 형태소 안에 공백이 있던 드문 토큰은 버린다."""
    toks = [tuple(x.split("\t")) for x in s.split(" ")] if s else []
    return [t for t in toks if len(t) == 2]


def fig_rates(tr, va):
    """아동 집단별 '알림'·'확인 필요' 비율 (train / val 나란히)."""
    import matplotlib.pyplot as plt
    from viz_features import BLUE, INK2, SURFACE, YELLOW, title

    groups = [("위기단계", ["정상군", "관찰필요", "상담필요", "학대의심", "응급"]),
              ("학대의심", ["(해당 없음)", "방임", "정서학대", "성학대", "신체학대"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for ax, (col, order) in zip(axes, groups):
        labels, alert, check = [], [], []
        for g in order:
            for name, d in [("train", tr), ("val", va)]:
                s = d.loc[d[col] == g, "감지"]
                labels.append(f"{g}  ({name} {len(s)}명)" if name == "train" else f"(val {len(s)}명)")
                alert.append((s == "알림").mean() * 100)
                check.append((s == "확인 필요").mean() * 100)
        y = list(range(len(labels)))[::-1]
        ax.barh(y, alert, color=BLUE, height=0.7, edgecolor=SURFACE, linewidth=1.5, label="알림")
        ax.barh(y, check, left=alert, color=YELLOW, height=0.7, edgecolor=SURFACE, linewidth=1.5,
                label="확인 필요")
        for yi, v in zip(y, alert):
            if v >= 8:
                ax.text(v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
            else:
                ax.text(v + 0.5, yi, f"{v:.0f}%", ha="left", va="center", fontsize=8.5, color=INK2,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.5))
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("아동 비율 (%)")
        ax.set_title(f"임상가 판정: {col}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, loc="upper right")
    title(fig, "고백 감지 규칙은 학대 아동에게서 주로 울린다 — 정상군 3%, 신체학대 97%",
          "아동 1명당 상담 대답 전체에서 가장 높은 단계 · 위 막대 train, 아래 막대 val(규칙을 고칠 때 보지 않은 아동)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_5_1_disclosure_rates.png", dpi=130)
    plt.close(fig)


def main():
    a = pd.read_parquet(TABLES / "answers_tok.parquet", columns=["split", "ID", "영역", "항목", "text", "prev_q"])
    cache = pd.read_parquet(TABLES / "tok_cache.parquet")
    assert (a["text"].values == cache["text"].values).all()
    c = pd.read_parquet(TABLES / "children.parquet", columns=["split", "ID", "위기단계", "학대의심", "유형구분"])

    det = Detector()
    rows = []
    for i, (t, q) in enumerate(zip(cache["tok"], cache["q_tok"])):
        for r in det.detect(None, toks=dec(t), context_toks=[dec(q)]):
            rows.append({"row": i, **r})
        if i % 30000 == 0:
            print(f"{i:,} / {len(a):,}")
    hits = pd.DataFrame(rows)
    hits = hits.join(a[["split", "ID", "영역", "항목", "text", "prev_q"]], on="row")
    hits.to_parquet(TABLES / "disclosure_hits.parquet", index=False)

    # 아동 단위: 가장 높은 단계
    per = hits.groupby(["split", "ID"])["level"].agg(lambda s: summarize([{"level": x} for x in s]))
    c = c.merge(per.rename("감지").reset_index(), on=["split", "ID"], how="left")
    c["감지"] = c["감지"].fillna("없음")
    tr = c[c.split == "train"]

    def rate(col):
        t = pd.crosstab(tr[col], tr["감지"], normalize="index").mul(100).round(1)
        t["아동 수"] = tr[col].value_counts()
        return t[[x for x in ["알림", "확인 필요", "없음"] if x in t] + ["아동 수"]]

    by_level = rate("위기단계").reindex(["정상군", "관찰필요", "상담필요", "학대의심", "응급"])
    by_abuse = rate("학대의심")
    # 규칙은 train 표본을 보며 고쳤으므로, 손대지 않은 val로 같은 수치를 확인한다
    va = c[c.split == "val"]
    val_level = pd.crosstab(va["위기단계"], va["감지"], normalize="index").mul(100).round(1) \
        .reindex(["정상군", "관찰필요", "상담필요", "학대의심", "응급"])
    val_level["아동 수"] = va["위기단계"].value_counts()
    fig_rates(tr, va)
    cat = pd.crosstab(hits["category"], hits["level"])
    lines = [
        "# 고백 감지: 상담 데이터 점검",
        "",
        f"- 아이 대답 {len(a):,}개 중 감지된 문장 {len(hits):,}개 (알림 {(hits.level == '알림').sum():,}, "
        f"확인 필요 {(hits.level == '확인 필요').sum():,})",
        "- 앞 문맥 = 바로 앞 상담사 질문. 아동 단위로 가장 높은 단계를 셈. 표는 train 2,876명, 행 비율 %",
        "",
        "## 위기단계별",
        "```\n" + by_level.to_string() + "\n```",
        "",
        "## 임상가 학대의심 유형별",
        "```\n" + by_abuse.to_string() + "\n```",
        "",
        "## 위기단계별 — val 360명 (규칙을 고칠 때 보지 않은 아동)",
        "```\n" + val_level.to_string() + "\n```",
        "",
        "## 분류별 감지 문장 수",
        "```\n" + cat.to_string() + "\n```",
    ]
    (REPORTS / "disclosure_eval.md").write_text("\n".join(lines), encoding="utf-8")

    # 검토용 표본: 정상군 아이의 알림(오경보 후보) / 학대의심 아이의 알림
    h = hits.merge(c[["split", "ID", "위기단계"]], on=["split", "ID"])
    out = ["# 고백 감지 검토 표본", ""]
    for title, sub in [("정상군 아동에게서 나온 '알림' (오경보 후보)", h[(h.위기단계 == "정상군") & (h.level == "알림")]),
                       ("학대의심 아동에게서 나온 '알림'", h[(h.위기단계 == "학대의심") & (h.level == "알림")]),
                       ("'확인 필요'", h[h.level == "확인 필요"])]:
        out += [f"## {title} — 전체 {len(sub):,}개 중 30개", ""]
        for r in sub.sample(min(30, len(sub)), random_state=0).itertuples():
            out.append(f"- [{r.category} | {r.agent}→{r.victim}] Q: {r.prev_q[:50]}  \n  **A: {r.text[:120]}**")
        out.append("")
    (REPORTS / "disclosure_samples.md").write_text("\n".join(out), encoding="utf-8")
    print("saved reports/disclosure_eval.md, reports/disclosure_samples.md")


if __name__ == "__main__":
    main()
