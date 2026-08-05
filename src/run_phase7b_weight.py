# -*- coding: utf-8 -*-
"""Phase 7-b — 학습 가중치를 **지표 구조에 정렬**시킨다.

━━ 문제 ━━
지표:  Score = 영업장 평균( 품목 평균( 유효날짜 평균 SMAPE ) )
       → 한 칸의 실제 무게 = 1/(영업장 9개) × 1/(그 영업장 품목수) × 1/(그 품목 유효날짜수)

학습:  모든 행을 **똑같이** 취급한다.
       → 자주 팔리는 품목, 품목이 많은 영업장이 행 수만큼 과대 학습된다.

예) 담하는 42품목, 화담숲카페는 5품목.
    지표에서는 화담숲카페 품목 하나가 담하 품목 하나보다 **8.4배** 중요한데,
    학습에서는 담하 쪽 행이 훨씬 많아 **정반대**로 배운다.

━━ 해법 ━━
    w_row = 1 / ( 그 품목이 속한 영업장의 품목수 × 그 품목의 학습행 수 )
→ 각 품목이 총합 동일한 무게, 각 영업장이 총합 동일한 무게를 갖는다. **지표와 정확히 같은 구조.**

전면 적용이 과할 수 있으므로 **강도 α** 로 보간한다 (자유도 1, 과적합 위험 최소):
    w(α) = w^α          α=0 → 현행(균등) · α=1 → 지표와 완전 정렬

채점은 두 가지로 본다:
    (a) 보정 없이  — 가중치 효과만 분리
    (b) v9 보정 얹어  — 실제 파이프라인
둘 다에서 좋아져야 진짜다. 판정은 **4폴드 일관성 + 2σ**.
"""
import os
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F
import validate as V
from metrics import competition_score, make_weights

SEEDS = (2024, 913)
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FLOOR = 1.0
SIGMA2 = 0.0032
NT = os.cpu_count()
V9_BND, V9_C = [2.0, 10.0], [0.55, 0.90, 1.02]

PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    fn_all = F.feature_names()
    keep = [i for i, k in enumerate(fn_all) if k not in set(F.PROF_KEYS)]
    names = [fn_all[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]

    stores = np.array(ctx.store_of_item)
    n_item_in_store = np.array([int((stores == s).sum()) for s in stores])

    print("=" * 96)
    print("Phase 7-b — 학습 가중치를 지표 구조에 정렬")
    print("=" * 96)
    print("  [영업장별 품목 수와 지표상 품목 1개의 무게]")
    seen = {}
    for s in stores:
        seen.setdefault(s, int((stores == s).sum()))
    mn = min(seen.values())
    for s, n in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<14s} {n:>3d}품목   품목 하나의 무게 {min(seen.values())/n:>5.2f}"
              f"  ({'기준' if n == max(seen.values()) else f'{max(seen.values())/n:.1f}배'})")
    print()

    folds = []
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        it = mtr[m, 2]                                  # 학습행의 품목 인덱스
        cnt = np.bincount(it, minlength=ctx.n).astype(np.float64)
        # w ∝ 1/(영업장 품목수 × 그 품목의 학습행 수)
        w_item = np.divide(1.0, n_item_in_store * np.maximum(cnt, 1),
                           out=np.zeros(ctx.n), where=cnt > 0)
        w = w_item[it]
        w /= w.mean()
        folds.append(dict(name=fname,
                          Xt=np.ascontiguousarray(Xtr[m][:, keep]),
                          yt=np.log1p(np.maximum(ytr[m], 1.0)), w=w,
                          Xv=np.ascontiguousarray(Xva[:, keep]),
                          y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        q = np.percentile(w, [1, 25, 50, 75, 99])
        print(f"  [{fname:<8s}] 학습 {len(w):>7,}행 · 가중치 분위 "
              f"1%={q[0]:.2f} 25%={q[1]:.2f} 50%={q[2]:.2f} "
              f"75%={q[3]:.2f} 99%={q[4]:.2f}  (최대/최소 {w.max()/w.min():.0f}배)")
    print()

    def sc(d, p, calib):
        q = p
        if calib:
            idx = np.digitize(q, V9_BND)
            q = q * np.asarray(V9_C)[idx]
        return competition_score(d["y"], np.maximum(q, FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    res = {}
    print("=" * 96)
    print("학습 — 강도 α 별")
    print("=" * 96)
    for a in ALPHAS:
        t1 = time.time()
        raw, cal = [], []
        for d in folds:
            ww = d["w"] ** a
            ww = ww / ww.mean()
            ds = lgb.Dataset(d["Xt"], label=d["yt"], weight=ww,
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            p = np.mean([np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                            num_boost_round=ROUNDS).predict(d["Xv"]))
                         for sd in SEEDS], 0)
            raw.append(sc(d, p, False))
            cal.append(sc(d, p, True))
        res[a] = dict(raw=raw, cal=cal)
        print(f"  α={a:.2f}  보정없이 {np.mean(raw):.4f} · v9보정 {np.mean(cal):.4f}"
              f"   ({time.time()-t1:.0f}s)", flush=True)

    for tag, lab in (("raw", "(a) 보정 없이 — 가중치 효과만"),
                     ("cal", "(b) v9 보정 얹어 — 실제 파이프라인")):
        print("\n" + "=" * 96)
        print(lab)
        print("=" * 96)
        b = res[0.0][tag]
        print(f"  {'α':>6s}" + "".join(f"{d['name']:>11s}" for d in folds)
              + f"{'평균':>10s}{'현행대비':>10s}{'일관':>6s}")
        for a in ALPHAS:
            s = res[a][tag]
            ok = all(s[i] < b[i] for i in range(4))
            mark = "  ←현행" if a == 0.0 else ""
            print(f"  {a:>6.2f}" + "".join(f"{x:>11.4f}" for x in s)
                  + f"{np.mean(s):>10.4f}{np.mean(b)-np.mean(s):>+10.4f}"
                  + f"{'○' if ok else '×':>5s}{mark}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    ok_any = False
    for tag, lab in (("raw", "보정 없이"), ("cal", "v9 보정 얹어")):
        b = res[0.0][tag]
        ba = min((a for a in ALPHAS if a > 0),
                 key=lambda a: float(np.mean(res[a][tag])))
        g = float(np.mean(b)) - float(np.mean(res[ba][tag]))
        cons = all(res[ba][tag][i] < b[i] for i in range(4))
        print(f"  {lab:<14s} 최적 α={ba:.2f}  개선 {g:+.4f}  4폴드 일관 "
              f"{'○' if cons else '×'}")
        ok_any |= (g > SIGMA2 and cons)
    print()
    if ok_any:
        print("  → ★ 채택 후보. 전체학습 후 제출로 확인.")
    else:
        print("  → 2σ 미달 또는 폴드 불일치. 현행(균등 가중) 유지.")

    json.dump({str(a): res[a] for a in ALPHAS},
              open(os.path.join(C.EXPERIMENTS, "phase7b_weight.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase7b_weight.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
