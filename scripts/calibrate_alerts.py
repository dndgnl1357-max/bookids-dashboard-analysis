"""AI 관찰 규칙의 기준값을 모의실험으로 정한다.

사용: .venv/Scripts/python scripts/calibrate_alerts.py
출력: reports/alert_calibration.md, reports/fig_6_1_alert_tradeoff.png, lexicon/base_rates.json

아직 Bookids 주간 기록이 없으므로 다음 가정으로 감정 낱말 횟수를 만든다 (실제 기록이 생기면 다시 돌린다).
- 기준 비율: 상담 데이터 정상군 아이가 스스로 꺼낸 감정 낱말 비율 (어절당)
- 아이마다 평소 수준이 다르다            : 아이 배수 ~ Gamma(2, 0.5)
- 그림책마다 분위기가 다르다 (무서운 책)   : 책 배수 ~ Gamma(k, 1/k), k=1.0  (클수록 책 차이 작음)
- 한 주에 2~3권, 한 권에 약 60어절
- '변화'가 있는 아이: 5주차부터 모든 책에서 감정 낱말이 effect배
판정: 오경보율 = 변화 없는 아이-주 중 알림 비율, 감지율 = 변화 후 2주 안에 한 번이라도 알림
"""
import json
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from story_metrics import ALERT, alert_test
from viz_features import BLUE, GRAY, INK2, ORANGE, SURFACE, title

ROOT = Path(__file__).resolve().parents[1]
TABLES, REPORTS, LEX = ROOT / "data" / "tables", ROOT / "reports", ROOT / "lexicon"
RNG = np.random.default_rng(0)
N_CHILD, N_WEEK, CHANGE_WEEK = 3000, 6, 4  # 0부터 세서 4 = 5주차
BOOK_K = 1.0
STORY_RATE = 0.02  # 이야기 속 부정 감정 낱말 비율 가정 (2회/100어절). Bookids 기록이 생기면 실제 값으로


def base_rates():
    f = pd.read_parquet(TABLES / "child_features.parquet")
    normal = f[(f.split == "train") & (f["위기단계"] == "정상군")]
    rates = {e: float(normal[f"emo_{e}_self_p100"].mean() / 100)
             for e in ["기쁨", "놀람", "무서움", "슬픔", "화남"]}
    (LEX / "base_rates.json").write_text(json.dumps(rates, ensure_ascii=False, indent=1), encoding="utf-8")
    return rates


def scan_rates(rule, rates_per100=(0.16, 0.5, 1.0, 2.0, 4.0), effects=(2.0, 3.0)):
    """이야기 속 감정 낱말 비율과 집계 기간(1주/2주)에 따라 오경보율·감지율이 어떻게 바뀌는지."""
    global RATE
    rows = []
    for r100 in rates_per100:
        RATE = r100 / 100
        null = simulate(RATE, 1.0)
        alts = {e: simulate(RATE, e) for e in effects}
        for win in (1, 2):
            row = {"비율(회/100어절)": r100, "기간": f"{win}주",
                   "오경보율": run_rule(*null, **rule, window=win).mean()}
            for e, cw in alts.items():
                row[f"감지율 x{e:g}"] = run_rule(*cw, **rule, window=win).any(1).mean()
            rows.append(row)
            print(row)
    return pd.DataFrame(rows)


def simulate(rate, effect):
    """[아이, 주, 책] 단위 (횟수, 어절)."""
    n_books = RNG.integers(2, 4, size=(N_CHILD, N_WEEK))
    child_mult = RNG.gamma(2.0, 0.5, size=(N_CHILD, 1, 1))
    words = np.clip(RNG.normal(60, 20, size=(N_CHILD, N_WEEK, 3)), 20, None)
    book_mult = RNG.gamma(BOOK_K, 1 / BOOK_K, size=(N_CHILD, N_WEEK, 3))
    week_eff = np.where(np.arange(N_WEEK) >= CHANGE_WEEK, effect, 1.0)[None, :, None]
    lam = rate * child_mult * book_mult * week_eff * words
    counts = RNG.poisson(lam)
    present = np.arange(3)[None, None, :] < n_books[..., None]
    return counts * present, words * present


def run_rule(counts, words, alpha, min_count, min_books, prior, window=1):
    """변화 시작 후 2주 동안의 판정 [아이, 판정 횟수].
    window=1: 4주차, 5주차를 각각 그 주만 보고 판정. window=2: 5주차에 4~5주차 2주를 묶어 판정.
    기준 기간은 판정 기간 바로 앞 최대 4주."""
    ends = [CHANGE_WEEK, CHANGE_WEEK + 1] if window == 1 else [CHANGE_WEEK + 1]
    out = np.zeros((counts.shape[0], len(ends)), dtype=bool)
    for wi, w in enumerate(ends):
        cur = slice(w - window + 1, w + 1)
        base = slice(max(0, w - window + 1 - 4), w - window + 1)
        x, n = counts[:, cur].sum((1, 2)), words[:, cur].sum((1, 2))
        X, N = counts[:, base].sum((1, 2)), words[:, base].sum((1, 2))
        books = (counts[:, cur] > 0).sum((1, 2))
        n_books = (words[:, cur] > 0).sum((1, 2))
        base_books = (counts[:, base] > 0).sum((1, 2))
        base_n = (words[:, base] > 0).sum((1, 2))
        # 퍼짐 조건: 최근 이야기의 60% 이상, 그 전보다 높은 비율 (story_metrics와 같은 규칙)
        spread = (books >= min_books) & (books / np.maximum(n_books, 1) >= ALERT["min_story_share"]) & \
                 (books / np.maximum(n_books, 1) > base_books / np.maximum(base_n, 1))
        for i in range(counts.shape[0]):
            if x[i] >= min_count and spread[i]:
                p, _ = alert_test(x[i], n[i], X[i], N[i], RATE, n_books[i], prior, BOOK_K)
                out[i, wi] = p < alpha
    return out


def fig_scan(scan):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for win, color in [("1주", GRAY), ("2주", BLUE)]:
        d = scan[scan["기간"] == win]
        x = d["비율(회/100어절)"]
        ax.plot(x, d["감지율 x3"] * 100, "-o", color=color, linewidth=2, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"{win} 묶어 판정 · 3배 증가 감지율")
        ax.plot(x, d["오경보율"] * 100, "--", color=color, linewidth=1.5, label=f"{win} 묶어 판정 · 오경보율")
    ax.axvline(0.16, color=INK2, linewidth=1, linestyle=":")
    ax.text(0.17, 92, "상담 데이터의\n무서움 비율 0.16", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([0.16, 0.5, 1, 2, 4], ["0.16", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("이야기 속 감정 낱말 비율 (회/100어절, 로그 눈금)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="center left")
    title(fig, "감정 낱말이 드물면 주 단위 변화는 잡을 수 없다 — 2주 묶음이 낫다",
          "선택한 규칙 기준 · 한 주 2~3권 × 약 60어절 가정 · 실제 이야기 속 비율은 Bookids 기록이 쌓이면 확인해야 함")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_6_2_rate_scan.png", dpi=130)
    plt.close(fig)


def main():
    global RATE
    rates = base_rates()
    emo = "무서움"
    # 규칙 비교는 '이야기 속에서 그럴 법한' 비율(2회/100어절)에서 한다.
    # 상담 데이터 비율(0.16)에서는 어떤 규칙도 감지율이 0에 가까워 비교가 안 된다 (아래 둘째 표 참고)
    RATE = STORY_RATE
    grid = list(product([0.05, 0.01, 0.001], [3, 4, 5], [2, ALERT["min_books"]], [300]))
    null_c, null_w = simulate(RATE, 1.0)
    alt = {eff: simulate(RATE, eff) for eff in [2.0, 3.0, 5.0]}

    rows = []
    win = ALERT["window_weeks"]
    for alpha, mc, mb, prior in grid:
        fa = run_rule(null_c, null_w, alpha, mc, mb, prior, window=win).mean()
        row = {"alpha": alpha, "min_count": mc, "min_books": mb, "오경보율(아이-주)": fa}
        for eff, (c, w) in alt.items():
            row[f"감지율 x{eff:g}"] = run_rule(c, w, alpha, mc, mb, prior, window=win).any(1).mean()
        rows.append(row)
        print(row)
    res = pd.DataFrame(rows)
    chosen = (res["alpha"] == ALERT["alpha"]) & (res["min_count"] == ALERT["min_count"]) & \
             (res["min_books"] == ALERT["min_books"])

    # 그림: 오경보율 vs 감지율(x3)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for mb, color, label in [(2, GRAY, "이야기 2편 이상"), (ALERT["min_books"], BLUE, f"이야기 {ALERT['min_books']}편 이상")]:
        d = res[res["min_books"] == mb]
        ax.scatter(d["오경보율(아이-주)"] * 100, d["감지율 x3"] * 100, s=70, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for alpha, d in res[res["min_books"] == ALERT["min_books"]].groupby("alpha"):
        r = d.iloc[0]
        ax.annotate(f"유의수준 {alpha:g}", (r["오경보율(아이-주)"] * 100, r["감지율 x3"] * 100),
                    textcoords="offset points", xytext=(10, -16), fontsize=9, color=INK2)
    c = res[chosen].iloc[0]
    ax.scatter(c["오경보율(아이-주)"] * 100, c["감지율 x3"] * 100, s=260, facecolor="none",
               edgecolor=ORANGE, linewidth=2.5, zorder=4)
    ax.annotate(f"선택한 규칙\n오경보 {c['오경보율(아이-주)']:.1%}, 감지 {c['감지율 x3']:.0%}",
                (c["오경보율(아이-주)"] * 100, c["감지율 x3"] * 100), textcoords="offset points",
                xytext=(14, 18), fontsize=10, color=INK2)
    ax.axvline(1, color=INK2, linewidth=1, linestyle="--")
    ax.text(1.03, 92, "오경보 1% 선", fontsize=9, color=INK2)
    ax.set_xlabel("오경보율: 변화 없는 아이가 판정 한 번에 알림을 받을 확률 (%)")
    ax.set_ylabel("감지율: 3배로 는 아이를 찾을 확률 (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.85))
    title(fig, "AI 관찰 규칙 정하기 — 오경보를 줄이면 감지도 줄어든다",
          f"모의실험 아이 {N_CHILD:,}명 · 이야기 속 감정 낱말 {STORY_RATE * 100:g}회/100어절 가정 · 최근 {win}주 묶음 판정 · "
          "점 하나 = 규칙 하나 (유의수준 × 최소 횟수 × 책 수)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(REPORTS / "fig_6_1_alert_tradeoff.png", dpi=130)
    plt.close(fig)

    rule = {"alpha": ALERT["alpha"], "min_count": ALERT["min_count"], "min_books": ALERT["min_books"],
            "prior": ALERT["prior_words"]}
    scan = scan_rates(rule)
    fig_scan(scan)

    fmt = res.copy()
    for col in fmt.columns[3:]:
        fmt[col] = (fmt[col] * 100).round(1)
    sfmt = scan.copy()
    for col in sfmt.columns[2:]:
        sfmt[col] = (sfmt[col] * 100).round(1)
    lines = [
        "# AI 관찰 규칙 기준값 모의실험",
        "",
        f"- 첫 표: 규칙 비교. 이야기 속 감정 낱말 {STORY_RATE * 100:g}회/100어절 가정, 최근 {win}주 묶음 판정",
        f"- 아이 {N_CHILD:,}명 × 6주, 한 주 2~3권 × 약 60어절, 이야기마다 분위기 차이 Gamma(k={BOOK_K})",
        "- 단위 %. 오경보율 = 변화 없는 아이의 판정 중 알림, 감지율 = 5주차부터 effect배가 된 아이를 찾은 비율",
        f"- 선택: alpha={ALERT['alpha']}, min_count={ALERT['min_count']}, min_books={ALERT['min_books']}, "
        f"최근 {ALERT['window_weeks']}주 묶음 vs 그 전 {ALERT['baseline_weeks']}주",
        "",
        "```\n" + fmt.to_string(index=False) + "\n```",
        "",
        "기준 비율: " + ", ".join(f"{e} {r * 100:.2f}" for e, r in rates.items()) + " (회/100어절)",
        "",
        "## 이야기 속 감정 낱말 비율 × 집계 기간 (선택한 규칙). 0.16 = 상담 데이터 무서움 비율",
        "```\n" + sfmt.to_string(index=False) + "\n```",
    ]
    (REPORTS / "alert_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved reports/alert_calibration.md, fig_6_1_alert_tradeoff.png")


if __name__ == "__main__":
    main()
