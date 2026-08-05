# -*- coding: utf-8 -*-
"""Phase 11-b — 피처 제거 사다리. **사전등록된 순서로만** 훑는다.

    python run_phase11b_ladder.py [시드수]        (기본 8)

━━ 왜 '사다리' 이고 왜 '사전등록' 인가 ━━
9-e 에서 실력차가 0인 설계 11개를 훑었더니 최댓값이 자동으로 평균+1.6σ 로 나왔다.
설정을 많이 보고 최댓값을 고르는 순간 편향이 들어온다. 그래서:
  · 후보 목록을 **실험 전에 고정**하고 도중에 바꾸지 않는다
  · **최댓값이 아니라 '단조/고원' 모양**을 본다
  · 제출은 **가장 넓은 고원의 중앙 한 지점만**

━━ 무엇을 자르는가 (Phase 11-a 의 구조 분석에서 나온 것만) ━━
A. 증명된 잉여   `d_last`  — `sd_lastweek` 와 **비트 단위로 동일**. 정보 손실 0
B. 낡는 열       `month · doy_sin · doy_cos · day` — 절대 날짜를 지목한다.
                 Phase 5-e: log2(잎 수)와 먼 미래 열화폭의 상관 +0.982.
                 9-d: 레벨↑ → gain↑ → 성적↓. **gain 은 '암기 통로의 크기'를 재기도 한다.**
                 cal 은 지금까지 통째로만 잘라봤고(+0.0173) 그 안을 가른 적이 없다.
C. 중복 덩어리   창 통계 11개가 r>=0.9 로 한 덩어리. 그중 상관 최상위 3개를 뺀다.

━━ 판정 ━━
· 채점은 **v17 확정 후처리**(t1=1.8 · c2=0.90 · c3=1.02 · 하한 1.0 · 기하 스냅)를 통과시킨 뒤.
  후처리가 정수 격자라 원본의 미세한 변화는 묻힌다 — 그게 실제 제출 조건이므로 그대로 잰다.
· **모든 사다리 칸이 같은 시드 집합**을 쓴다(짝비교).
· 채택 기준: 4폴드 평균 개선 > 2σ̂ **그리고** 4폴드 중 3개 이상 같은 방향.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights, seg_snap

NT = os.cpu_count()
ALL_SEEDS = (42, 7, 2024, 913, 31, 101, 202, 303, 404, 505,
             606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515)

PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]

ABS_TIME = ["month", "doy_sin", "doy_cos", "day"]
WIN_DUP = ["w_posmedian", "w_last14", "w_std"]     # 11-a 상관 최상위 3

# ★ 사전등록된 사다리 — 실행 전에 고정. 도중에 바꾸지 않는다.
RUNGS = [
    ("R0 현행 57",              []),
    ("R1 −d_last",              ["d_last"]),
    ("R2 R1−month",             ["d_last", "month"]),
    ("R3 R2−doy(2)",            ["d_last", "month", "doy_sin", "doy_cos"]),
    ("R4 R3−day = 절대시간 전부", ["d_last"] + ABS_TIME),
    ("R5 R1−창중복3",           ["d_last"] + WIN_DUP),
    ("R6 R4+R5 결합",           ["d_last"] + ABS_TIME + WIN_DUP),
]


def main():
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seeds = ALL_SEEDS[:nseed]
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    base_keep, base_names = F.active_columns(), F.active_names()

    print("=" * 100)
    print(f"Phase 11-b — 피처 제거 사다리 · 시드 {nseed}개 · 사다리 {len(RUNGS)}칸")
    print("=" * 100)
    for lab, drop in RUNGS:
        print(f"  {lab:<26s} 피처 {len(base_names)-len(drop):>2d}개"
              + (f"  (−{', '.join(drop)})" if drop else ""))
    print(flush=True)

    # 폴드별 샘플을 한 번만 만들어 재사용 (사다리 칸마다 열만 슬라이스)
    prepared = []
    for fname, cut, v0, v1_ in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1_, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        prepared.append(dict(
            name=fname,
            Xt=np.ascontiguousarray(Xtr[m][:, base_keep]),
            yt=np.log1p(np.maximum(ytr[m], 1.0)),
            Xv=np.ascontiguousarray(Xva[:, base_keep]),
            y=yva, W=W, valid=valid))
        del Xtr, Xva
        print(f"  준비 [{fname}] 학습 {prepared[-1]['Xt'].shape[0]:,}행 · "
              f"검증 {len(yva):,}칸", flush=True)

    def score(f, raw):
        p = seg_snap(raw)                       # v17 확정 후처리
        a = np.abs(f["y"]); q = np.abs(p); den = a + q
        t = np.zeros(len(a)); m = f["valid"] & (den > 0)
        t[m] = 2.0 * np.abs(f["y"][m] - p[m]) / den[m]
        return float((f["W"] * t).sum())

    results = {}
    for lab, drop in RUNGS:
        t1 = time.time()
        sub = [i for i, n in enumerate(base_names) if n not in drop]
        names = [base_names[i] for i in sub]
        cats = [c for c in F.CATEGORICAL if c in names]
        per_fold = []
        for f in prepared:
            Xt = np.ascontiguousarray(f["Xt"][:, sub])
            Xv = np.ascontiguousarray(f["Xv"][:, sub])
            ps = []
            for sd in seeds:
                ds = lgb.Dataset(Xt, label=f["yt"], feature_name=names,
                                 categorical_feature=cats, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            per_fold.append(score(f, np.mean(ps, 0)))
            del Xt, Xv
        results[lab] = per_fold
        b = results["R0 현행 57"]
        print(f"  [{lab:<26s}] " + " ".join(f"{x:.5f}" for x in per_fold)
              + f"  평균 {np.mean(per_fold):.5f}"
              + (f"  기준대비 {np.mean(b)-np.mean(per_fold):+.5f}"
                 f"  일관 {sum(x<y for x,y in zip(per_fold,b))}/4"
                 if lab != "R0 현행 57" else "")
              + f"   ({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 100)
    print(f"  {'사다리':<26s}" + "".join(f"{f['name']:>11s}" for f in prepared)
          + f"{'평균':>11s}{'기준대비':>11s}{'일관':>7s}")
    b = results["R0 현행 57"]
    for lab, _ in RUNGS:
        v = results[lab]
        print(f"  {lab:<26s}" + "".join(f"{x:>11.5f}" for x in v)
              + f"{np.mean(v):>11.5f}"
              + (f"{np.mean(b)-np.mean(v):>+11.5f}"
                 f"{sum(x<y for x,y in zip(v,b)):>5d}/4" if lab != RUNGS[0][0]
                 else f"{'—':>11s}{'—':>7s}"))
    print("\n  판정: 단조 또는 고원이면 후보. 특정 한 칸만 튀면 노이즈(규칙 ⑦).")
    print(f"  문턱: 시드 {nseed}개 기준 2σ̂ ≈ {2*0.00321/np.sqrt(nseed):.5f} (10-d 의 σ₁=0.00321 기준)")

    json.dump(dict(seeds=list(seeds), rungs=[[l, d] for l, d in RUNGS],
                   results={k: [float(x) for x in v] for k, v in results.items()},
                   minutes=(time.time() - t0) / 60),
              open(os.path.join(C.EXPERIMENTS, "phase11b_ladder.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
