"""3단계 시각화: 감정 낱말과 위기단계를 그림으로.

사용: .venv/Scripts/python scripts/viz_features.py  (features.py 실행 후)
출력: reports/fig_3_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
LEX = ROOT / "lexicon"
REPORTS = ROOT / "reports"

# 색: dataviz 기본 팔레트
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b5b4ad"
# 위기단계는 순서가 있으므로 파랑 한 색의 밝기 단계로 (합계점수 평균 순)
LEVELS = ["정상군", "관찰필요", "상담필요", "학대의심", "응급"]
LEVEL_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
HIGH_RISK = {"응급", "학대의심"}

FEATURE_NAMES = {
    "mean_eojeol": "대답 길이 (평균 어절)",
    "one_word_ratio": "한 어절 대답 비율",
    "unsure_ratio": "'모르겠어요' 비율",
    "neg_p100": "부정 표현 (안/못/없)",
    "first_person_p100": "1인칭 (나/저)",
    "family_p100": "가족 낱말",
    "emo_negated_p100": "부정된 감정 ('안 행복해요')",
}

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10,
})


def title(fig, main, sub):
    fig.text(0.01, 0.98, main, fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.925, sub, fontsize=10, color=INK2, va="top")


def feature_label(col, emotions):
    if col in FEATURE_NAMES:
        return FEATURE_NAMES[col]
    for e in emotions:
        if col == f"emo_{e}_self_p100":
            return f"{e} 낱말 (스스로 꺼낸 것)"
        if col == f"emo_{e}_p100":
            return f"{e} 낱말 (전체)"
    return col


def fig_emotion_by_level(tr, emotions, dashboard):
    """감정 분류별로, 위기단계마다 100어절당 평균 횟수."""
    n = len(emotions)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(13, 6.6), sharey=False)
    for ax, e in zip(axes.flat, emotions):
        m = tr.groupby("위기단계")[f"emo_{e}_p100"].mean().reindex(LEVELS)
        ax.bar(range(5), m.values, color=LEVEL_COLORS, width=0.72, edgecolor=SURFACE, linewidth=2)
        tag = "" if e in dashboard else "  (예측용)"
        ax.set_title(f"{e}{tag}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks(range(5), LEVELS, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        # 양 끝(정상군, 응급)에만 값 표시
        for i in (0, 4):
            v = m.values[i]
            ax.text(i, v, f"{v:.1f}" if v >= 0.1 else f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK2)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    title(fig, "위기단계가 높을수록 기쁨 낱말은 줄고, 무서움·슬픔·화남 낱말은 늘어난다",
          "100어절당 평균 횟수 · 부정('안 행복해요')과 바람('~면 좋겠어요')은 제외 · train 2,876명 · 막대 색이 진할수록 위기단계가 높음")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_3_1_emotion_by_level.png", dpi=130)
    plt.close(fig)


def fig_echo(hits, emotions):
    """감정 낱말이 어떻게 나왔는지: 스스로 / 질문 따라 / 부정 / 바람."""
    h = hits.copy()
    h["유형"] = np.select([h["바람"], h["부정"], h["메아리"]],
                         ["바람", "부정", "질문 따라 함"], "스스로 꺼냄")
    order = ["스스로 꺼냄", "질문 따라 함", "부정", "바람"]
    colors = [BLUE, GRAY, ORANGE, YELLOW]
    pct = (pd.crosstab(h["감정"], h["유형"], normalize="index") * 100).reindex(
        index=emotions[::-1], columns=order).fillna(0)
    counts = h["감정"].value_counts()

    fig, ax = plt.subplots(figsize=(11, 5.4))
    left = np.zeros(len(pct))
    for col, color in zip(order, colors):
        ax.barh(pct.index, pct[col], left=left, color=color, height=0.62,
                edgecolor=SURFACE, linewidth=2, label=col)
        for i, (l, v) in enumerate(zip(left, pct[col])):
            if v >= 7:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if color in (BLUE, ORANGE) else INK)
        left += pct[col].values
    ax.set_yticks(range(len(pct)), [f"{e}  ({counts.get(e, 0):,}회)" for e in pct.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
    title(fig, "감정 낱말의 상당수는 아이가 스스로 꺼낸 것이 아니다",
          "화남은 절반, 슬픔은 약 40%가 바로 앞 질문의 낱말을 따라 말한 것 · Bookids(자유 발화)에는 '스스로 꺼냄'에 가까운 신호만 옮겨 갈 수 있다")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_3_2_echo.png", dpi=130)
    plt.close(fig)


def fig_separability(tr, emotions):
    """특징 하나로 정상군 vs 응급·학대의심을 얼마나 가를 수 있는지 (AUC)."""
    m = tr["위기단계"].isin(HIGH_RISK | {"정상군"})
    y = tr.loc[m, "위기단계"].isin(HIGH_RISK).astype(int)
    cols = list(FEATURE_NAMES) + [f"emo_{e}_self_p100" for e in emotions]
    rows = []
    for col in cols:
        auc = roc_auc_score(y, tr.loc[m, col])
        rows.append((feature_label(col, emotions), max(auc, 1 - auc), auc >= 0.5))
    df = pd.DataFrame(rows, columns=["특징", "구분력", "위험군에서 많음"]).sort_values("구분력")

    fig, ax = plt.subplots(figsize=(11, 6.4))
    colors = [ORANGE if up else BLUE for up in df["위험군에서 많음"]]
    ax.barh(df["특징"], df["구분력"] - 0.5, left=0.5, color=colors, height=0.66,
            edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(df["구분력"]):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.axvline(0.5, color=INK2, linewidth=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("구분력 (AUC) · 0.5 = 동전 던지기, 1.0 = 완벽")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="위험군에서 더 많음"),
                       Patch(color=BLUE, label="위험군에서 더 적음")],
              loc="lower right", frameon=False)
    title(fig, "특징 하나만으로 정상군과 위험군(응급·학대의심)을 얼마나 가를 수 있나",
          "train 정상군 598명 vs 응급·학대의심 1,163명 · 감정은 '스스로 꺼낸 것' 기준")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_3_3_separability.png", dpi=130)
    plt.close(fig)


def fig_overlap(tr):
    """가장 강한 특징(기쁨)도 개인 단위로는 겹친다."""
    col = "emo_기쁨_self_p100"
    normal = tr.loc[tr["위기단계"] == "정상군", col]
    risk = tr.loc[tr["위기단계"].isin(HIGH_RISK), col]
    bins = np.linspace(0, np.percentile(tr[col], 99), 30)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.hist(normal, bins=bins, density=True, color=BLUE, alpha=0.55, label=f"정상군 ({len(normal)}명)")
    ax.hist(risk, bins=bins, density=True, color=ORANGE, alpha=0.55, label=f"응급·학대의심 ({len(risk)}명)")
    for s, c in [(normal, BLUE), (risk, ORANGE)]:
        ax.axvline(s.median(), color=c, linewidth=2, linestyle="--")
    ax.set_xlabel("아이가 스스로 꺼낸 기쁨 낱말 (100어절당)")
    ax.set_ylabel("비율")
    ax.legend(frameon=False)
    title(fig, "가장 강한 신호(기쁨 낱말)도 0~3 구간에서는 두 집단이 겹친다",
          "점선 = 중앙값 · 집단 차이는 뚜렷하지만, 기쁨 낱말이 적다는 것만으로 한 아이를 위험군이라 볼 수는 없다")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_3_4_overlap.png", dpi=130)
    plt.close(fig)


def main():
    f = pd.read_parquet(TABLES / "child_features.parquet")
    hits = pd.read_parquet(TABLES / "emotion_hits.parquet")
    lex = pd.read_csv(LEX / "emotion_lexicon.csv")
    emotions = list(dict.fromkeys(lex["감정"]))
    dashboard = set(lex.loc[lex["분류"] == "대시보드", "감정"])
    tr = f[f["split"] == "train"]

    fig_emotion_by_level(tr, emotions, dashboard)
    fig_echo(hits, emotions)
    fig_separability(tr, emotions)
    fig_overlap(tr)
    print("saved reports/fig_3_*.png")


if __name__ == "__main__":
    main()
