"""3단계: 아이 대답 형태소 분석 -> 감정 낱말 / 말하기 특징 -> 위기단계와의 관련성.

사용: .venv/Scripts/python scripts/features.py
입력: data/tables/utterances.parquet, children.parquet, lexicon/emotion_lexicon.csv
출력:
  data/tables/answers_tok.parquet   대답별 형태소
  data/tables/child_features.parquet 아동별 특징
  lexicon/emotion_review.csv        감정 사전 검토용 (빈도, 메아리 비율, 예문)
  lexicon/candidates.csv            사전에 없는 자주 나온 낱말 (추가 후보)
  reports/features.md               특징과 위기단계의 관련성
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
LEX = ROOT / "lexicon"
REPORTS = ROOT / "reports"

CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "XR", "MAG"}
NEGATION = {("안", "MAG"), ("못", "MAG"), ("아니", "VCN"), ("없", "VA"), ("않", "VX"), ("말", "VX")}
FIRST_PERSON = {"나", "저", "내", "제", "우리", "저희"}
FAMILY = {"엄마", "아빠", "어머니", "아버지", "할머니", "할아버지", "언니", "오빠", "누나", "형", "동생"}
UNSURE = ("모르", "기억")  # '모르겠어요', '기억 안 나요'

# 위험군 비교: 정상군 vs 응급+학대의심
HIGH_RISK = {"응급", "학대의심"}


def base_tag(tag):
    return tag.split("-")[0]  # 'VA-I'(불규칙) -> 'VA'


def is_negated(toks, j):
    """감정 낱말 toks[j]가 부정됐는지: '안 좋아요', '즐겁지 않아요', '걱정 없어요', '화 난 적 없어요'."""
    if any(t in {("안", "MAG"), ("못", "MAG")} for t in toks[max(0, j - 2):j]):
        return True
    for t in toks[j + 1:j + 6]:
        if t in {("않", "VX"), ("없", "VA"), ("아니", "VCN")}:
            return True
        if t[1] in {"EF", "SF"} or (t[1] == "EC" and t[0] not in {"지", "게", "어", "아"}):
            return False  # 절이 끝나면 부정 범위도 끝
    return False


def is_wish(toks, j):
    """'때렸으면 좋겠어요'처럼 바람을 나타내는 '좋'."""
    return toks[j][0] == "좋" and any(t == ("겠", "EP") for t in toks[j + 1:j + 3])


def tokenize(kiwi, texts):
    out = []
    for toks in kiwi.tokenize(texts):
        out.append([(t.form, base_tag(t.tag)) for t in toks])
    return out


def main():
    u = pd.read_parquet(TABLES / "utterances.parquet")
    c = pd.read_parquet(TABLES / "children.parquet")
    lex = pd.read_csv(LEX / "emotion_lexicon.csv")
    lex_map = {(r.형태소, r.품사): r.감정 for r in lex.itertuples()}
    EMOTIONS = list(dict.fromkeys(lex["감정"]))  # 대시보드 5종 + 예측용 분류, CSV 순서대로

    u["text"] = u["text"].fillna("").str.strip()
    # 각 대답(A) 바로 앞 질문(Q)을 붙인다 (같은 아동, 같은 항목 안에서)
    u = u.sort_values(["split", "ID", "영역", "항목", "turn"]).reset_index(drop=True)
    prev_q = u["text"].where(u["type"] == "Q").groupby([u.split, u.ID, u.영역, u.항목]).ffill()
    u["prev_q"] = prev_q.where(u["type"] == "A")
    a = u[u["type"] == "A"].copy().reset_index(drop=True)
    a["prev_q"] = a["prev_q"].fillna("")

    # 1. 형태소 분석 (5분 남짓 걸려서 캐시한다. 사전만 고쳤을 때는 다시 돌리지 않음)
    cache = TABLES / "tok_cache.parquet"
    enc = lambda toks: " ".join(f"{f}\t{g}" for f, g in toks)
    dec = lambda s: [tuple(x.split("\t")) for x in s.split(" ")] if s else []
    if cache.exists() and (cached := pd.read_parquet(cache))["text"].tolist() == a["text"].tolist():
        print("using cached tokens")
        a["tok"] = cached["tok"].map(dec)
        q_tok = cached["q_tok"].map(dec).tolist()
    else:
        kiwi = Kiwi()
        print(f"tokenizing {len(a)} answers ...")
        a["tok"] = tokenize(kiwi, a["text"].tolist())
        print("tokenizing preceding questions ...")
        q_tok = tokenize(kiwi, a["prev_q"].tolist())
        pd.DataFrame({"text": a["text"], "tok": a["tok"].map(enc), "q_tok": [enc(t) for t in q_tok]}) \
            .to_parquet(cache, index=False)
    q_sets = [set(t) for t in q_tok]

    # 2. 대답별 특징
    a["eojeol"] = a["text"].str.split().str.len()
    a["n_morph"] = a["tok"].str.len()
    a["neg"] = a["tok"].map(lambda t: sum(x in NEGATION for x in t))
    a["first_person"] = a["tok"].map(lambda t: sum(f in FIRST_PERSON and g == "NP" for f, g in t))
    a["family"] = a["tok"].map(lambda t: sum(f in FAMILY for f, g in t))
    a["unsure"] = a["text"].str.contains("|".join(UNSURE))
    a["one_word"] = a["eojeol"] <= 1

    # 감정 낱말: 전체 / 질문에 없던 것(스스로 꺼낸 것). 부정·바람은 감정으로 세지 않고 따로 센다.
    review_rows = []
    counts = {f"emo_{e}{s}": np.zeros(len(a), dtype=int) for e in EMOTIONS for s in ("", "_self")}
    counts["emo_negated"] = np.zeros(len(a), dtype=int)
    for i, (toks, qs) in enumerate(zip(a["tok"], q_sets)):
        for j, t in enumerate(toks):
            emo = lex_map.get(t)
            if not emo:
                continue
            echoed = t in qs
            neg, wish = is_negated(toks, j), is_wish(toks, j)
            review_rows.append((emo, t[0], t[1], echoed, neg, wish, i))
            if wish:
                continue
            if neg:
                counts["emo_negated"][i] += 1
                continue
            counts[f"emo_{emo}"][i] += 1
            if not echoed:
                counts[f"emo_{emo}_self"][i] += 1
    for col, v in counts.items():
        a[col] = v

    # 3. 감정 사전 검토표
    rv = pd.DataFrame(review_rows, columns=["감정", "형태소", "품사", "메아리", "부정", "바람", "row"])
    review = rv.groupby(["감정", "형태소", "품사"]).agg(
        빈도=("row", "size"), 메아리비율=("메아리", "mean"), 부정비율=("부정", "mean"),
        바람비율=("바람", "mean"), rows=("row", list)).reset_index()
    for col in ["메아리비율", "부정비율", "바람비율"]:
        review[col] = (review[col] * 100).round(1)
    rv.to_parquet(TABLES / "emotion_hits.parquet", index=False)  # 시각화용
    review["예문"] = review["rows"].map(
        lambda rs: " | ".join(f"Q:{a.at[r, 'prev_q']} → A:{a.at[r, 'text']}" for r in rs[:3]))
    review = review.drop(columns="rows")
    # 사전에 있지만 한 번도 안 나온 낱말도 표시
    seen = set(zip(review.형태소, review.품사))
    missing = lex[[(m, p) not in seen for m, p in zip(lex.형태소, lex.품사)]]
    review = pd.concat([review, missing[["감정", "형태소", "품사"]].assign(빈도=0)], ignore_index=True)
    review["검토(유지/삭제/이동)"] = ""
    review.sort_values(["감정", "빈도"], ascending=[True, False]).to_csv(
        LEX / "emotion_review.csv", index=False, encoding="utf-8-sig")

    # 사전 추가 후보: 자주 나온 내용어 중 사전에 없는 것
    cnt = Counter(t for toks in a["tok"] for t in toks if t[1] in CONTENT_TAGS and t not in lex_map)
    pd.DataFrame([(f, g, n) for (f, g), n in cnt.most_common(400)],
                 columns=["형태소", "품사", "빈도"]).assign(감정="").to_csv(
        LEX / "candidates.csv", index=False, encoding="utf-8-sig")

    # 4. 아동별 특징
    emo_cols = [f"emo_{e}" for e in EMOTIONS] + [f"emo_{e}_self" for e in EMOTIONS] + ["emo_negated"]
    g = a.groupby(["split", "ID"])
    f = pd.DataFrame({
        "answers": g.size(),
        "words": g["eojeol"].sum(),
        "mean_eojeol": g["eojeol"].mean(),
        "one_word_ratio": g["one_word"].mean(),
        "unsure_ratio": g["unsure"].mean(),
        "ttr": g["tok"].agg(lambda s: len({t for toks in s for t in toks})
                            / max(1, sum(len(toks) for toks in s))),
    })
    per100 = 100 / f["words"]
    for col in ["neg", "first_person", "family"] + emo_cols:
        f[f"{col}_p100"] = g[col].sum() * per100  # 100어절당 횟수
    f = f.reset_index().merge(c[["split", "ID", "나이", "성별", "유형구분", "위기단계", "합계점수"]],
                              on=["split", "ID"])
    f.to_parquet(TABLES / "child_features.parquet", index=False)
    a.drop(columns=["tok"]).assign(tok=a["tok"].map(lambda t: " ".join(f"{x}/{y}" for x, y in t))) \
        .to_parquet(TABLES / "answers_tok.parquet", index=False)

    # 5. 위기단계와의 관련성 (train만 사용, val은 모델 평가용으로 남겨 둔다)
    tr = f[f.split == "train"]
    feats = [col for col in f.columns if col not in
             {"split", "ID", "나이", "성별", "유형구분", "위기단계", "합계점수", "answers", "words"}]
    rows = []
    bin_mask = tr["위기단계"].isin(HIGH_RISK | {"정상군"})
    y = tr.loc[bin_mask, "위기단계"].isin(HIGH_RISK).astype(int)
    for col in feats:
        groups = [tr.loc[tr["위기단계"] == lv, col].values for lv in sorted(tr["위기단계"].unique())]
        h, p = kruskal(*groups)
        eps2 = h / (len(tr) - 1)  # 효과크기: 0.01 작음, 0.06 중간, 0.14 큼
        auc = roc_auc_score(y, tr.loc[bin_mask, col])
        means = tr.groupby("위기단계")[col].mean()
        rows.append({"특징": col, "효과크기(ε²)": round(eps2, 3), "p": p,
                     "AUC(정상 vs 응급·학대의심)": round(auc, 3),
                     **{lv: round(v, 3) for lv, v in means.items()}})
    stats = pd.DataFrame(rows).sort_values("효과크기(ε²)", ascending=False)

    echo = rv.groupby("감정")["메아리"].mean().mul(100).round(1)
    lines = [
        "# 3단계: 발화 특징과 위기단계",
        "",
        f"- 분석 대상: 아이 대답 {len(a):,}개 (아동 {len(f):,}명), 통계는 train {len(tr):,}명만 사용",
        "- `_p100` = 100어절당 횟수, `_self` = 바로 앞 질문에 없던 감정 낱말(아이가 스스로 꺼낸 것)",
        "- 효과크기 ε²: 0.01 작음 / 0.06 중간 / 0.14 큼. AUC 0.5 = 구분 못함, 1.0 = 완벽",
        "",
        "## 감정 낱말이 질문을 따라 말한 것(메아리)인 비율 %",
        "```\n" + echo.to_string() + "\n```",
        "",
        "## 특징별 관련성 (효과크기 순)",
        "```\n" + stats.to_string(index=False, float_format=lambda v: f"{v:.3g}") + "\n```",
    ]
    (REPORTS / "features.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved reports/features.md, lexicon/emotion_review.csv, lexicon/candidates.csv")


if __name__ == "__main__":
    main()
