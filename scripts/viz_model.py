"""4단계 시각화: 모델 비교, 혼동행렬, 현장 비율에 따른 정밀도, 특징 계수.

사용: .venv/Scripts/python scripts/viz_model.py  (model.py 실행 후)
출력: reports/fig_4_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from viz_features import (BLUE, GRAY, GRID, INK, INK2, ORANGE, SURFACE, feature_label, title)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

MODEL_NAMES = {
    "F": "F  말하기 특징",
    "T": "T  대답 낱말",
    "F+T": "F+T  둘 다",
    "QA": "QA  질문+대답 (참고)",
}
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fig_compare(res):
    names = list(res["models"])
    f1 = [res["models"][n]["multi"]["val_f1_macro"] for n in names]
    auc = [res["models"][n]["binary"]["val_auc"] for n in names]
    colors = [GRAY if n == "QA" else BLUE for n in names]
    labels = [MODEL_NAMES[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, vals, name, base in [(axes[0], f1, "5단계 맞히기 (macro-F1)", 0.2),
                                 (axes[1], auc, "위험군 가려내기 (AUC)", 0.5)]:
        y = np.arange(len(names))[::-1]
        ax.barh(y, np.array(vals) - base, left=base, color=colors, height=0.62,
                edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=10, color=INK2)
        ax.set_yticks(y, labels)
        ax.set_xlim(base, 1.0)
        ax.axvline(base, color=INK2, linewidth=1)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("0.2 = 아무렇게나 찍기 (5개 중 1개)")
    axes[1].set_xlabel("0.5 = 동전 던지기, 1.0 = 완벽")
    title(fig, "검증용 360명에서의 성능",
          "회색 QA는 상담사 질문이 함께 있는 경우의 참고치 · Bookids에는 상담 질문이 없으므로 파랑 모델이 현실적인 기준")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_4_1_compare.png", dpi=130)
    plt.close(fig)


def fig_confusion(res, name):
    levels = res["levels"]
    cm = np.array(res["models"][name]["multi"]["confusion"], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(levels)):
        for j in range(len(levels)):
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=9.5, color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(levels)), levels)
    ax.set_xlabel("모델이 예측한 단계")
    ax.set_ylabel("실제 단계 (임상가 판정)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    title(fig, f"어디서 헷갈리나 — {name} 모델의 5단계 예측",
          "가로 한 줄 = 실제 그 단계인 아이들이 어떻게 예측됐는지 (줄마다 합 100%) · 대각선이 맞힌 것, 틀린 것은 대부분 바로 옆 단계")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_4_2_confusion.png", dpi=130)
    plt.close(fig)


def fig_ppv(res, names=("F", "F+T")):
    p = np.linspace(0.01, 0.40, 200)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for n, color in zip(names, [ORANGE, BLUE]):
        b = res["models"][n]["binary"]
        sens, spec = b["val_sensitivity"], b["val_specificity"]
        ppv = sens * p / (sens * p + (1 - spec) * (1 - p))
        ax.plot(p * 100, ppv * 100, color=color, linewidth=2,
                label=f"{MODEL_NAMES[n]} (민감도 {sens:.0%}, 특이도 {spec:.0%})")
        for x in (5, 10):
            v = np.interp(x / 100, p, ppv) * 100
            ax.plot(x, v, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(8, -14 if n == "F" else 6),
                        fontsize=9.5, color=INK2)
    ax.axvline(36, color=INK2, linewidth=1, linestyle="--")
    ax.text(36.4, 8, "이 데이터의\n위험군 비율 36%", fontsize=9, color=INK2)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 100)
    ax.set_xlabel("실제 현장의 위험군 비율 (%)")
    ax.set_ylabel("'위험' 알림 중 진짜 위험군 (%)")
    ax.legend(frameon=False, loc="upper left")
    title(fig, "위험군이 드문 현장에서는 '위험' 알림의 상당수가 오경보가 된다",
          "같은 모델이라도 현장 비율이 5%면 알림 중 진짜는 소수 · 이 데이터는 위험군을 일부러 많이 모은 것이라 그대로 쓰면 과대평가")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_4_3_ppv.png", dpi=130)
    plt.close(fig)


def fig_coef(res):
    coef = res["models"]["F"]["coef"]
    emotions = [c[4:-10] for c in coef if c.endswith("_self_p100")]
    items = sorted(coef.items(), key=lambda kv: kv[1])
    labels = [feature_label(k, emotions) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, vals, color=[ORANGE if v > 0 else BLUE for v in vals], height=0.66,
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("← 위험군 가능성을 낮춤        위험군 가능성을 높임 →")
    ax.grid(axis="y", visible=False)
    title(fig, "말하기 특징 모델(F)은 무엇을 보고 판단하나",
          "다른 특징이 같을 때, 그 특징이 평균보다 1표준편차 높으면 위험군 쪽으로 얼마나 기우는지 (로지스틱 회귀 계수)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_4_4_coef.png", dpi=130)
    plt.close(fig)


def main():
    res = json.loads((REPORTS / "model_results.json").read_text(encoding="utf-8"))
    fig_compare(res)
    fig_confusion(res, "F+T")
    fig_ppv(res)
    fig_coef(res)
    print("saved reports/fig_4_*.png")


if __name__ == "__main__":
    main()
