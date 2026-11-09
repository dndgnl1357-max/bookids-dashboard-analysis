"""분석 결과 페이지용 추가 그림.

사용: .venv/Scripts/python scripts/viz_report.py
출력:
  reports/fig_7_audio_pitch.png   녹음 점검: 상담사 vs '아이' 목소리 높이 (audio_check.md 표에서)
  reports/fig_8_dev_demo.png      발달 지표 6주 변화: 성장(D) · 감소(G) · 나머지 아이 (가상 반)
"""
import re
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dev_metrics as dev
from story_metrics import StoryAnalyzer
from viz_features import BLUE, GRAY, INK2, ORANGE, SURFACE, title

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def fig_audio():
    txt = (REPORTS / "audio_check.md").read_text(encoding="utf-8").split("```")[1].strip().splitlines()
    rows = [l.split() for l in txt[1:]]
    q = np.array([float(r[-2]) for r in rows])
    a = np.array([float(r[-1]) for r in rows])
    r = np.corrcoef(q, a)[0, 1]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhspan(220, 320, color=BLUE, alpha=0.07)
    ax.text(102, 312, "7~12세 아이 목소리의 일반적인 범위", fontsize=9, color=INK2, va="top")
    male = q < 160
    ax.scatter(q[~male], a[~male], s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="상담사 목소리가 높은 파일", zorder=3)
    ax.scatter(q[male], a[male], s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label="상담사 목소리가 낮은 파일", zorder=3)
    ax.plot([100, 330], [100, 330], color=INK2, linewidth=1, linestyle="--")
    ax.text(300, 290, "상담사 = 아이", fontsize=9, color=INK2, rotation=33)
    ax.set_xlim(100, 330)
    ax.set_ylim(100, 330)
    ax.set_xlabel("상담사 질문의 목소리 높이 (Hz, 파일별 중앙값)")
    ax.set_ylabel("'아이' 대답의 목소리 높이 (Hz)")
    ax.legend(frameon=False, loc="lower right")
    title(fig, f"'아이' 목소리가 상담사 목소리를 따라간다 — 상관 {r:.2f}",
          f"무작위 {len(q)}개 파일 · 실제 상담이라면 상담사의 목소리와 아이 목소리는 관계가 없어야 함 · 상담사가 낮은 목소리면 '아이'도 성인 남성 범위")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_7_audio_pitch.png", dpi=130)
    plt.close(fig)


def fig_dev():
    log = pd.read_csv(ROOT / "data" / "demo" / "story_log.csv")
    an = StoryAnalyzer()
    a = an.analyze(log)
    weeks = sorted(a["week"].unique())
    keys = ["mlc_m", "sg_score", "conn_rate"]
    vals = {}
    for (child, wk), rows in a.groupby(["child_id", "week"]):
        books = an._books(rows)
        for k in keys:
            vals[(child, wk, k)] = dev.METRICS[k][2](books)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    labels = ["6주 전", "5주 전", "4주 전", "3주 전", "지난주", "이번 주"]
    for ax, k in zip(axes, keys):
        label, unit = dev.METRICS[k][0], dev.METRICS[k][1]
        for child in log["child_id"].unique():
            if child in ("아이 D", "아이 G"):
                continue
            ax.plot(range(6), [vals.get((child, w, k)) for w in weeks], color=GRAY, linewidth=1, alpha=0.6)
        for child, color, name in [("아이 D", BLUE, "D 성장"), ("아이 G", ORANGE, "G 감소")]:
            ys = [vals.get((child, w, k)) for w in weeks]
            ax.plot(range(6), ys, "-o", color=color, linewidth=2.5, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name, zorder=3)
        ax.set_xticks(range(6), labels, fontsize=8.5)
        ax.set_title(f"{label} ({unit})", loc="left", fontsize=11, fontweight="bold")
    axes[0].plot([], [], color=GRAY, linewidth=1, label="나머지 16명")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    title(fig, "발달 지표는 성장과 감소를 따라가지만, 주마다 흔들림이 크다",
          "가상 반 18명 × 6주 (실제 아이 말이 아님) · 그래서 ‘늘었어요/줄었어요’는 최근 3주와 그 전 3주를 비교할 때만 판정")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_8_dev_demo.png", dpi=130)
    plt.close(fig)


def fig_refined():
    """세부 지표: 그 전 3주 → 최근 3주, D(성장)·G(감소)·나머지 아이들."""
    import json
    d = json.loads((ROOT / "outputs" / "dashboard_week.json").read_text(encoding="utf-8"))
    items = [("intro_explicit_pct", "인물을 이름으로 소개 (%)"), ("complex_pct", "복문 비율 (%)"),
             ("pronouns_per_10_sent", "지칭어 ‘걔·그거’ (회/10문장)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    for ax, (key, label) in zip(axes[:3], items):
        for cid, r in d["reports"].items():
            a, b = r["development"]["refined"]["before_3w"], r["development"]["refined"]["recent_3w"]
            if not a or not b:
                continue
            color, lw, z = {"아이 D": (BLUE, 2.5, 3), "아이 G": (ORANGE, 2.5, 3)}.get(cid, (GRAY, 1, 1))
            ax.plot([0, 1], [a[key], b[key]], "-o", color=color, linewidth=lw, markersize=5 if z > 1 else 3,
                    alpha=1 if z > 1 else 0.6, zorder=z)
        ax.set_xticks([0, 1], ["그 전 3주", "최근 3주"])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(label, loc="left", fontsize=10.5, fontweight="bold")
    ax = axes[3]
    lv = ["나열·묘사", "불완전한 에피소드", "완전한 에피소드", "여러 에피소드"]
    colors = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]
    bars = []
    for cid in ["아이 D", "아이 G"]:
        for per, name in [("before_3w", "전"), ("recent_3w", "최근")]:
            bars.append((f"{cid[-1]} {name}", d["reports"][cid]["development"]["refined"][per]["episode_levels"]))
    for i, (name, levels) in enumerate(bars[::-1]):
        left, total = 0, sum(levels.values())
        for k, c in zip(lv, colors):
            w = levels[k] / total * 100
            ax.barh(i, w, left=left, color=c, edgecolor=SURFACE, linewidth=1.5, height=0.7)
            left += w
    ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("이야기 비율 (%)")
    ax.set_title("이야기 구조 단계", loc="left", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=k) for k, c in zip(lv, colors)], fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    axes[0].plot([], [], "-o", color=BLUE, label="D 성장")
    axes[0].plot([], [], "-o", color=ORANGE, label="G 감소")
    axes[0].plot([], [], color=GRAY, label="나머지 16명")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3)
    title(fig, "세부 지표도 같은 방향으로 움직인다 — 성장한 아이는 인물을 이름으로 소개하고, 문장을 잇고, 이야기를 완성한다",
          "가상 반 18명 (실제 아이 말이 아님) · 판정 없이 값만 · 선 하나 = 아이 한 명")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_9_refined.png", dpi=130)
    plt.close(fig)


def fig_next_steps():
    """반 전체: 다음 단계 질문이 어떤 요소에 대해 몇 명에게 제안됐나."""
    import json
    from collections import Counter
    d = json.loads((ROOT / "outputs" / "dashboard_week.json").read_text(encoding="utf-8"))
    cnt = Counter(s["target"] for r in d["reports"].values() for s in r["development"]["next_steps"])
    tier = {s["target"]: s["tier"] for r in d["reports"].values() for s in r["development"]["next_steps"]}
    order = ["배경", "등장인물", "시작 사건", "결과", "시도", "목표", "반응", "원인 이음말", "인용", "생각을 나타내는 말"]
    q = {s["target"]: s["question"] for r in d["reports"].values() for s in r["development"]["next_steps"]}
    labels = [k for k in order if cnt.get(k)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, [cnt[k] for k in labels], color=[ORANGE if tier[k] == "먼저" else BLUE for k in labels], height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, k in zip(y, labels):
        ax.text(cnt[k] + 0.15, yi, f"{cnt[k]}명 · {tier[k]} · 예: “{q[k]}”", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(cnt.values()) + 7)
    ax.set_xlabel("이 질문을 제안받은 아이 수 (아이당 최대 2개)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", visible=False)
    title(fig, "반 전체로 보면, 어떤 이야기 요소를 함께 연습하면 좋을지 보인다",
          "가상 반 18명 · 주황 = 먼저 해 볼 질문(4~5세 대부분이 쓰는 요소가 또래보다 적을 때) · 파랑 = 한 단계 더(4~5세에 아직 드문 요소)")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(REPORTS / "fig_10_next_steps.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_audio()
    fig_dev()
    fig_refined()
    fig_next_steps()
    print("saved reports/fig_7_audio_pitch.png, fig_8_dev_demo.png, fig_9_refined.png, fig_10_next_steps.png")
