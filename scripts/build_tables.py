"""라벨 JSON -> 분석용 표 3개 (아동 / 항목 / 발화).

사용: .venv/Scripts/python scripts/build_tables.py
출력: data/tables/{children,items,utterances}.parquet (+ 확인용 csv)
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
OUT = ROOT / "data" / "tables"

# 항목마다 공통으로 있는 키. 나머지는 항목별 세부 필드(extra)로 묶는다.
ITEM_COMMON = {"항목", "임상가코멘트", "점수", "audio"}


def to_sec(ts):
    """'MM:SS.mmm' 또는 'HH:MM:SS.mmm' -> 초."""
    if not ts:
        return None
    sec = 0.0
    try:
        for part in str(ts).split(":"):
            sec = sec * 60 + float(part)
    except ValueError:  # '--' 같은 누락 표기
        return None
    return sec


def flatten_val(v):
    """{'val': ..., '기타': ...} 형태를 문자열 하나로."""
    if isinstance(v, dict):
        s = v.get("val", "")
        if v.get("기타"):
            s = f"{s} / 기타: {v['기타']}"
        return s
    return v


def parse(path, split):
    d = json.loads(path.read_text(encoding="utf-8"))
    info = d["info"]
    cid = str(info["ID"])

    child = {"split": split, "file": path.stem, **info, "ID": cid}
    items, utts = [], []

    for dom in d["list"]:
        name = dom["문항"]
        child[f"{name}_합계"] = dom.get("문항합계")
        child[f"{name}_위기단계"] = dom.get("위기단계")

        for it in dom["list"]:
            extra = {k: flatten_val(v) for k, v in it.items() if k not in ITEM_COMMON}
            items.append({
                "split": split, "ID": cid, "영역": name, "항목": it.get("항목"),
                "점수": it.get("점수"),
                "임상가코멘트": flatten_val(it.get("임상가코멘트")),
                "세부": json.dumps(extra, ensure_ascii=False),
                "발화수": len(it.get("audio", [])),
            })
            for turn, a in enumerate(it.get("audio", [])):
                start, end = to_sec(a.get("start")), to_sec(a.get("end"))
                utts.append({
                    "split": split, "ID": cid, "영역": name, "항목": it.get("항목"),
                    "turn": turn, "type": a.get("type"), "text": a.get("text"),
                    "start": start, "end": end,
                    "dur": (end - start) if start is not None and end is not None else None,
                })
    return child, items, utts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    children, items, utts, errors = [], [], [], []

    for split in ("train", "val"):
        for p in sorted((LABELS / split).glob("*.json")):
            try:
                c, i, u = parse(p, split)
            except Exception as e:  # 형식이 어긋난 파일은 기록만 하고 넘어간다
                errors.append((split, p.name, repr(e)))
                continue
            children.append(c)
            items.extend(i)
            utts.extend(u)

    tables = {
        "children": pd.DataFrame(children),
        "items": pd.DataFrame(items),
        "utterances": pd.DataFrame(utts),
    }

    # 숫자 필드에 int와 str이 섞여 있다 (예: 나이 9 / "9"). 숫자로 통일하고 변환 실패는 기록한다.
    numeric = {
        "children": ["나이", "합계점수", "행동특성 점수"]
                    + [c for c in tables["children"].columns if c.endswith("_합계")],
        "items": ["점수"],
    }
    for name, cols in numeric.items():
        df = tables[name]
        for col in cols:
            if col not in df.columns:
                continue
            conv = pd.to_numeric(df[col], errors="coerce")
            bad = df.loc[conv.isna() & df[col].notna(), col]
            if len(bad):
                errors.extend((name, col, f"숫자 아님: {v!r}") for v in bad.unique())
            df[col] = conv
    # 나머지 object 열은 문자열로 통일 (parquet 저장용)
    for df in tables.values():
        for col in df.columns[df.dtypes == object]:
            df[col] = df[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {df.shape}")

    err_path = OUT / "parse_errors.csv"
    if errors:
        pd.DataFrame(errors, columns=["split", "file", "error"]).to_csv(
            err_path, index=False, encoding="utf-8-sig")
    else:
        err_path.unlink(missing_ok=True)
    print(f"errors: {len(errors)}")


if __name__ == "__main__":
    main()
