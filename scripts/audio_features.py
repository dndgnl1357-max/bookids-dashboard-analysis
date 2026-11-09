"""음성 실험 1단계: mp3에서 발화 구간별 음향 특징을 뽑는다. 중간에 멈춰도 이어서 돌릴 수 있다.

사용:
  .venv/Scripts/python scripts/audio_features.py --check 3     # 3개 파일로 시간 정렬 점검만
  .venv/Scripts/python scripts/audio_features.py --workers 6   # 전체 (이미 끝난 아동은 건너뜀)
출력:
  data/audio_feats/{split}_{ID}.parquet   아동 1명의 발화별 특징 (이 파일이 있으면 다시 하지 않음)
  reports/audio_run.log                   진행 기록

mp3는 압축을 풀지 않고 zip에서 바로 읽는다. 특징 (Praat/parselmouth):
  f0_med, f0_sd_st (반음), f0_range_st(10~90% 범위, 반음), voiced_ratio  목소리 높낮이와 변화
  int_mean, int_sd (dB)                                                 크기와 변화
  pause_ratio                                                           구간 안의 쉼 비율
  hnr                                                                   목소리 맑기 (조화 대 잡음비)
"""
import argparse
import io
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "024.아동·청소년 상담 데이터" / "3.개방데이터" / "1.데이터"
ZIPS = {"train": DATA / "Training" / "01.원천데이터" / "TS_in.zip",
        "val": DATA / "Validation" / "01.원천데이터" / "VS_in.zip"}
OUT = ROOT / "data" / "audio_feats"
LOG = ROOT / "reports" / "audio_run.log"
TABLES = ROOT / "data" / "tables"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def seg_features(snd):
    """parselmouth.Sound 구간 하나의 특징."""
    if snd.get_total_duration() < 0.2:
        return {}
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"]
    voiced = f0[f0 > 0]
    inten = snd.to_intensity(time_step=0.01).values[0]
    out = {"voiced_ratio": float(len(voiced) / max(len(f0), 1))}
    if len(voiced) >= 5:
        st = 12 * np.log2(voiced / 100)
        out.update(f0_med=float(np.median(voiced)), f0_sd_st=float(np.std(st)),
                   f0_range_st=float(np.percentile(st, 90) - np.percentile(st, 10)))
    if len(inten):
        out.update(int_mean=float(np.mean(inten)), int_sd=float(np.std(inten)),
                   pause_ratio=float(np.mean(inten < inten.max() - 25)))
    try:
        hnr = snd.to_harmonicity(time_step=0.01).values[0]
        hnr = hnr[hnr > -200]
        if len(hnr):
            out["hnr"] = float(np.mean(hnr))
    except Exception:
        pass
    return out


def load_audio(zf, name):
    data, sr = sf.read(io.BytesIO(zf.read(name)), dtype="float32", always_2d=True)
    return parselmouth.Sound(data.mean(axis=1), sampling_frequency=sr)


def process(job):
    split, cid, name, segs = job
    path = OUT / f"{split}_{cid}.parquet"
    if path.exists():
        return cid, "skip", 0
    t0 = time.time()
    try:
        with zipfile.ZipFile(ZIPS[split]) as zf:
            snd = load_audio(zf, name)
    except Exception as e:
        return cid, f"error: {e!r}", 0
    total = snd.get_total_duration()
    rows = []
    for r in segs.itertuples():
        feats = {}
        if r.start is not None and r.end is not None and 0 <= r.start < r.end <= total + 0.5:
            feats = seg_features(snd.extract_part(r.start, min(r.end, total), preserve_times=False))
        rows.append({"split": split, "ID": cid, "uid": r.uid, "type": r.type, "start": r.start,
                     "end": r.end, "file_dur": total, **feats})
    pd.DataFrame(rows).to_parquet(path, index=False)
    return cid, "ok", time.time() - t0


def jobs():
    u = pd.read_parquet(TABLES / "utterances.parquet", columns=["split", "ID", "type", "start", "end"])
    u["uid"] = u.index
    out = []
    for split, zp in ZIPS.items():
        with zipfile.ZipFile(zp) as zf:
            names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".mp3")}
        for cid, segs in u[u.split == split].groupby("ID"):
            if cid in names:
                out.append((split, cid, names[cid], segs))
    return out


def check(n):
    """녹음 점검 (무작위 n개): 파일 길이 vs 마지막 타임스탬프, 발화 사이 소리 크기(실제 녹음이면 방 소음이 있음),
    대답(아이) 목소리 높이 (7~12세는 보통 220~300Hz, 어른 남성 100~150Hz)."""
    c = pd.read_parquet(TABLES / "children.parquet", columns=["split", "ID", "성별", "나이"])
    js = jobs()
    pick = np.random.default_rng(0).choice(len(js), size=min(n, len(js)), replace=False)
    rows = []
    for split, cid, name, segs in [js[i] for i in pick]:
        with zipfile.ZipFile(ZIPS[split]) as zf:
            snd = load_audio(zf, name)
        inten = snd.to_intensity(time_step=0.01)
        times, vals = inten.xs(), inten.values[0]
        inside = np.zeros_like(vals, dtype=bool)
        for r in segs.itertuples():
            if r.start is not None and r.end is not None:
                inside |= (times >= r.start) & (times <= r.end)
        f0 = {t: [] for t in "QA"}
        for r in segs.itertuples():
            if r.start is not None and r.end is not None and r.end <= snd.get_total_duration():
                ft = seg_features(snd.extract_part(r.start, r.end, preserve_times=False))
                if "f0_med" in ft:
                    f0[r.type].append(ft["f0_med"])
        gap = vals[~inside]
        info = c[(c.split == split) & (c.ID == cid)].iloc[0]
        rows.append({"split": split, "ID": cid, "성별": info["성별"], "나이": info["나이"],
                     "표본율": int(snd.sampling_frequency),
                     "길이차(초)": round(snd.get_total_duration() - segs["end"].max(), 1),
                     "발화 안 dB": round(float(vals[inside].mean()), 1),
                     "발화 사이 dB 중앙값": round(float(np.median(gap)), 1) if len(gap) else None,
                     "발화 사이 무음 비율": round(float(np.mean(gap < 10)), 2) if len(gap) else None,
                     "질문 F0": round(float(np.median(f0["Q"])), 0) if f0["Q"] else None,
                     "대답 F0": round(float(np.median(f0["A"])), 0) if f0["A"] else None})
        log(f"[check] {split} {cid} done")
    df = pd.DataFrame(rows)
    a = df["대답 F0"]
    lines = ["# 음성 녹음 점검 (무작위 표본)", "",
             f"- 표본 {len(df)}개. '발화 사이 무음 비율' = 발화 구간 밖에서 10dB 미만(사실상 소리 없음)인 비율",
             f"- 대답(아이) 목소리 높이: 160Hz 미만 {int((a < 160).sum())}개, 160~220Hz {int(a.between(160, 220).sum())}개, "
             f"220Hz 이상 {int((a > 220).sum())}개",
             f"- 발화 사이가 거의 무음(비율 0.9 이상)인 파일: {int((df['발화 사이 무음 비율'] >= 0.9).sum())}개",
             "", "```\n" + df.sort_values("대답 F0").to_string(index=False) + "\n```",
             "", "## 성별별 대답 F0", "```\n" + df.groupby("성별")["대답 F0"].describe().round(0).to_string() + "\n```"]
    (ROOT / "reports" / "audio_check.md").write_text("\n".join(lines), encoding="utf-8")
    log("[check] saved reports/audio_check.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.check:
        check(args.check)
        return
    js = jobs()
    todo = [j for j in js if not (OUT / f"{j[0]}_{j[1]}.parquet").exists()]
    log(f"start: 전체 {len(js)}명, 남은 {len(todo)}명, workers={args.workers}")
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for cid, status, sec in pool.imap_unordered(process, todo, chunksize=2):
            done += 1
            if status.startswith("error"):
                log(f"{cid} {status}")
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                log(f"진행 {done}/{len(todo)} · 경과 {el / 60:.1f}분 · 남은 예상 {el / done * (len(todo) - done) / 60:.0f}분")
    log("done")


if __name__ == "__main__":
    main()
