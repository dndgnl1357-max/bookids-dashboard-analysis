"""4단계: 위기단계 예측 기본 모델.

사용: .venv/Scripts/python scripts/model.py  (features.py 실행 후)
입력: data/tables/child_features.parquet, answers_tok.parquet, tok_cache.parquet, lexicon/emotion_lexicon.csv
출력:
  models/*.joblib                  학습된 모델
  data/tables/val_predictions.parquet  val 예측값
  reports/model_results.json        시각화용 수치
  reports/model.md                  결과 요약

과제
  - 5단계: 정상군 / 관찰필요 / 상담필요 / 학대의심 / 응급
  - 위험군 이진: 응급·학대의심 vs 나머지
평가: train에서 5겹 교차검증으로 하이퍼파라미터와 기준점을 고르고, val(360명)에서 한 번만 평가한다.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
LEX = ROOT / "lexicon"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

LEVELS = ["정상군", "관찰필요", "상담필요", "학대의심", "응급"]
HIGH_RISK = {"응급", "학대의심"}
TARGET_SENS = 0.80          # 위험군 이진: 위험군의 80%는 놓치지 않는 기준점
PREVALENCES = [0.02, 0.05, 0.10, 0.20]  # 실제 현장의 위험군 비율 가정 (데이터는 36%로 맞춰져 있음)
SPEECH = ["mean_eojeol", "one_word_ratio", "unsure_ratio", "neg_p100",
          "first_person_p100", "family_p100", "emo_negated_p100"]
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def split_tokens(s):
    return s.split()


def build_docs(f):
    """아동별 문서: 대답 형태소만(A) / 질문+대답 형태소(QA)."""
    a = pd.read_parquet(TABLES / "answers_tok.parquet", columns=["split", "ID", "text", "tok"])
    cache = pd.read_parquet(TABLES / "tok_cache.parquet", columns=["text", "q_tok"])
    assert (a["text"].values == cache["text"].values).all(), "tok_cache와 answers_tok 순서가 다름"
    a["q"] = cache["q_tok"].str.replace("\t", "/").str.split().map(
        lambda ts: " ".join("Q:" + t for t in ts) if isinstance(ts, list) else "")
    g = a.groupby(["split", "ID"])
    docs = pd.DataFrame({
        "doc_a": g["tok"].agg(" ".join),
        "doc_qa": (a["q"] + " " + a["tok"]).groupby([a.split, a.ID]).agg(" ".join),
    }).reset_index()
    return f.merge(docs, on=["split", "ID"])


def make_models(feat_cols):
    tfidf = lambda: TfidfVectorizer(tokenizer=split_tokens, token_pattern=None, lowercase=False,
                                    ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)
    lr = lambda: LogisticRegression(max_iter=3000)
    grid = {"clf__C": [0.03, 0.1, 0.3, 1, 3, 10]}
    return {
        "F": (Pipeline([("x", ColumnTransformer([("num", StandardScaler(), feat_cols)])),
                        ("clf", lr())]), grid),
        "T": (Pipeline([("x", ColumnTransformer([("txt", tfidf(), "doc_a")])), ("clf", lr())]), grid),
        "F+T": (Pipeline([("x", ColumnTransformer([("num", StandardScaler(), feat_cols),
                                                   ("txt", tfidf(), "doc_a")])),
                          ("clf", lr())]), grid),
        "QA": (Pipeline([("x", ColumnTransformer([("txt", tfidf(), "doc_qa")])), ("clf", lr())]), grid),
    }


def ppv_at(sens, spec, p):
    """실제 위험군 비율이 p일 때, 모델이 위험이라고 한 아이 중 진짜 위험군 비율."""
    return sens * p / (sens * p + (1 - spec) * (1 - p))


def main():
    MODELS.mkdir(exist_ok=True)
    lex = pd.read_csv(LEX / "emotion_lexicon.csv")
    model_emotions = list(dict.fromkeys(lex.loc[lex["모델"] == "Y", "감정"]))
    feat_cols = [f"emo_{e}_self_p100" for e in model_emotions] + SPEECH

    f = build_docs(pd.read_parquet(TABLES / "child_features.parquet"))
    tr, va = f[f.split == "train"].reset_index(drop=True), f[f.split == "val"].reset_index(drop=True)
    y5_tr, y5_va = tr["위기단계"], va["위기단계"]
    yb_tr, yb_va = tr["위기단계"].isin(HIGH_RISK).astype(int), va["위기단계"].isin(HIGH_RISK).astype(int)

    results = {"feat_cols": feat_cols, "levels": LEVELS, "prevalences": PREVALENCES,
               "n_train": len(tr), "n_val": len(va), "models": {}}
    preds = va[["ID", "위기단계"]].copy()

    for name, (pipe, grid) in make_models(feat_cols).items():
        print(f"== {name}")
        r = {}

        # 5단계
        gs = GridSearchCV(pipe, grid, cv=CV, scoring="f1_macro", n_jobs=-1).fit(tr, y5_tr)
        p5 = gs.predict(va)
        r["multi"] = {
            "C": gs.best_params_["clf__C"], "cv_f1_macro": gs.best_score_,
            "val_f1_macro": f1_score(y5_va, p5, average="macro"),
            "val_accuracy": accuracy_score(y5_va, p5),
            "confusion": confusion_matrix(y5_va, p5, labels=LEVELS).tolist(),
        }
        preds[f"{name}_5단계"] = p5
        joblib.dump(gs.best_estimator_, MODELS / f"{name}_multi.joblib")

        # 위험군 이진
        gs = GridSearchCV(pipe, grid, cv=CV, scoring="roc_auc", n_jobs=-1).fit(tr, yb_tr)
        best = gs.best_estimator_
        # 기준점: train 교차검증 확률에서 민감도 80%가 되는 값
        oof = cross_val_predict(best, tr, yb_tr, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        fpr, tpr, thr = roc_curve(yb_tr, oof)
        cut = float(thr[np.argmax(tpr >= TARGET_SENS)])
        prob = best.predict_proba(va)[:, 1]
        hit = prob >= cut
        sens = float(hit[yb_va == 1].mean())
        spec = float((~hit)[yb_va == 0].mean())
        fpr_v, tpr_v, _ = roc_curve(yb_va, prob)
        r["binary"] = {
            "C": gs.best_params_["clf__C"], "cv_auc": gs.best_score_,
            "val_auc": roc_auc_score(yb_va, prob), "threshold": cut,
            "val_sensitivity": sens, "val_specificity": spec,
            "ppv": {str(p): ppv_at(sens, spec, p) for p in PREVALENCES},
            "roc": {"fpr": fpr_v.tolist(), "tpr": tpr_v.tolist()},
        }
        preds[f"{name}_위험확률"] = prob
        joblib.dump(best, MODELS / f"{name}_binary.joblib")

        # 해석: F 모델 계수 (표준화된 특징 1단위당 위험군 로그오즈)
        if name == "F":
            coef = best.named_steps["clf"].coef_[0]
            r["coef"] = dict(zip(feat_cols, coef.tolist()))

        results["models"][name] = r
        print(f"   5단계 macro-F1 {r['multi']['val_f1_macro']:.3f} | 위험군 AUC {r['binary']['val_auc']:.3f}"
              f" | 민감도 {sens:.2f} 특이도 {spec:.2f}")

    preds.to_parquet(TABLES / "val_predictions.parquet", index=False)
    (REPORTS / "model_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                                encoding="utf-8")

    # 요약 표
    rows = []
    for name, r in results["models"].items():
        rows.append({
            "모델": name,
            "5단계 정확도": r["multi"]["val_accuracy"], "5단계 macro-F1": r["multi"]["val_f1_macro"],
            "위험군 AUC": r["binary"]["val_auc"],
            "민감도": r["binary"]["val_sensitivity"], "특이도": r["binary"]["val_specificity"],
            **{f"정밀도@{int(p*100)}%": r["binary"]["ppv"][str(p)] for p in PREVALENCES},
        })
    table = pd.DataFrame(rows).set_index("모델").round(3)
    lines = [
        "# 4단계: 위기단계 예측 기본 모델",
        "",
        f"- train {len(tr):,}명으로 학습(5겹 교차검증), val {len(va):,}명으로 평가",
        "- F = 말하기 특징 (스스로 꺼낸 감정 낱말 + 말하기 지표), T = 아이 대답 낱말(TF-IDF), "
        "F+T = 둘 다, QA = 질문+대답 낱말 (상담사 질문이 있는 경우의 참고치)",
        f"- 위험군 = 응급·학대의심. 기준점은 위험군의 {int(TARGET_SENS*100)}%를 찾도록 train에서 정함",
        "- 정밀도@p% = 실제 현장의 위험군 비율이 p%라면, 모델이 '위험'이라고 한 아이 중 진짜 위험군 비율",
        "",
        "```\n" + table.to_string() + "\n```",
    ]
    (REPORTS / "model.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved reports/model.md, reports/model_results.json")


if __name__ == "__main__":
    main()
