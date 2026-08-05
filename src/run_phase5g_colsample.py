# -*- coding: utf-8 -*-
"""Phase 5-g — 열 샘플링(colsample)이 진짜 레버인가. 후보 2개 제출본 생성.

배경: CV 저울이 두 번 연속 빗나갔다.
  - F2+F3(가까운 거리) 기준 → v4를 1위로 뽑았는데 LB 꼴찌
  - FAR(먼 거리) 기준 → leaves15가 +0.0179라 했는데 LB 동률
    (원인: 거리를 벌리려 학습량을 37~58%로 줄여서 F1과 같은 결함이 생김)

그런데 **두 저울이 같은 방향을 가리키는 축이 하나 있다 — 열 샘플링.**
  · Public 1위 v3(XGBoost)의 특징이 `colsample_bytree=0.6`
  · FAR 사다리에서도 `ff0.65`가 잎 개수와 **독립적으로** +0.0084 (S2 0.5463 → S9 0.5379)

그래서 **변수를 열 샘플링 하나로 고정**하고 LB로 직접 잰다.
  A : v1 그대로 + ff 0.85 → 0.65     ← 변수 하나만 바꾼 순수 실험
  B : leaves127 + ff0.65             ← 511은 나빴고 15는 무효였으니 중간 용량

동시에 **두 저울 모두에서 점수를 기록**한다. LB 결과가 나오면
"어느 저울이 LB를 예측했는가"를 처음으로 답할 수 있게 된다 — 지금 가장 알고 싶은 것.
(v1 기준선은 이미 측정돼 있다: 가까운 0.4709 / FAR 0.5483)
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

SEEDS = (2024, 913, 31)                 # 기존 '확인 시드' — v1 기준선과 같은 조건
SUBMIT_SEEDS = (42, 7, 2024, 913, 31)
FLOOR = 1.0
NT = os.cpu_count()

# (이름, 학습 종료일, 검증 시작, 검증 끝, 저울 종류)
FOLDS = [
    ("F2 겨울",   "2023-11-24", "2023-11-24", "2024-02-22", "가까움"),
    ("F3 봄",     "2024-02-23", "2024-02-23", "2024-06-08", "가까움"),
    ("FAR-봄",    "2023-11-24", "2024-02-23", "2024-06-08", "멂"),
    ("FAR-겨울",  "2023-08-25", "2023-11-24", "2024-02-22", "멂"),
]

COMMON = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              min_data_in_leaf=40, bagging_fraction=0.85, bagging_freq=1,
              lambda_l2=1.0, verbosity=-1, num_threads=NT)
CANDS = [
    ("A  leaves255+ff0.65", dict(num_leaves=255, feature_fraction=0.65), 1000),
    ("B  leaves127+ff0.65", dict(num_leaves=127, feature_fraction=0.65), 1000),
]
# 참고값 (이미 측정됨, 같은 시드): v1 = leaves255·ff0.85 → 가까움 0.4709 / 멂 0.5483
REF_V1 = dict(near=0.4709, far=0.5483)


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

    print("=" * 94)
    print("Phase 5-g — 열 샘플링이 레버인가 · 두 저울 동시 측정")
    print("=" * 94)

    folds = []
    for fname, cut, v0, v1_, kind in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1_, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        folds.append(dict(name=fname, kind=kind,
                          Xt=np.ascontiguousarray(Xtr[m][:, keep]),
                          yt=np.log1p(np.maximum(ytr[m], 1.0)),
                          Xv=np.ascontiguousarray(Xva[:, keep]),
                          y=yva, iids=mva[:, 2]))
        del Xtr, Xva
        print(f"  [{fname:<9s}] {kind:<4s} 학습 {len(trn):>3d} origin "
              f"({folds[-1]['Xt'].shape[0]:>7,}행) → 검증 {v0}~{v1_}")
    print()

    def evaluate(over, rounds):
        p = dict(COMMON, **over)
        out = {}
        for fd in folds:
            ds = lgb.Dataset(fd["Xt"], label=fd["yt"], feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            pr = np.mean([np.expm1(lgb.train(dict(p, seed=sd), ds,
                                             num_boost_round=rounds).predict(fd["Xv"]))
                          for sd in SEEDS], 0)
            out[fd["name"]] = competition_score(
                fd["y"], np.maximum(pr, FLOOR), fd["iids"],
                ctx.store_of_item, make_weights(1.0), ctx.n)
        return out

    res = []
    hdr = "  " + f"{'후보':<20s}" + "".join(f"{f['name']:>11s}" for f in folds)
    print(hdr + f"{'가까움':>10s}{'멂':>10s}{'시간':>8s}")
    for label, over, rounds in CANDS:
        t1 = time.time()
        s = evaluate(over, rounds)
        near = float(np.mean([s[f["name"]] for f in folds if f["kind"] == "가까움"]))
        far = float(np.mean([s[f["name"]] for f in folds if f["kind"] == "멂"]))
        res.append(dict(label=label, over=over, rounds=rounds,
                        per_fold=s, near=near, far=far))
        print("  " + f"{label:<20s}" + "".join(f"{s[f['name']]:>11.4f}" for f in folds)
              + f"{near:>10.4f}{far:>10.4f}{time.time()-t1:>7.0f}s", flush=True)

    print("\n  " + f"{'v1 (leaves255·ff0.85)':<20s}" + " " * (11 * len(folds))
          + f"{REF_V1['near']:>10.4f}{REF_V1['far']:>10.4f}   ← 기준선(기측정)")
    print("\n  [기준선 대비 · 양수 = 좋아짐]")
    for r in res:
        print(f"   {r['label']:<20s} 가까움 {REF_V1['near']-r['near']:+.4f} · "
              f"멂 {REF_V1['far']-r['far']:+.4f}")
    print("\n  ※ 두 저울 모두 LB 예측에 실패한 전력이 있다. 여기 숫자는 **기록용**이고")
    print("     판정은 LB로 한다. 나중에 '어느 저울이 맞았나'를 되짚기 위한 데이터다.")

    # ------------------------------------------------------------ 제출본
    print("\n" + "=" * 94)
    print("전체 데이터 학습 → 제출본 2개")
    print("=" * 94)
    for f in folds:
        f["Xt"] = f["Xv"] = None
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    Xa, ya, _ = F.build_samples(mat, dates,
                                list(range(C.WINDOW - 1, nd - C.HORIZON)), ctx)
    m = ya != 0
    Xt = np.ascontiguousarray(Xa[m][:, keep])
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa
    print(f"  학습행 {Xt.shape[0]:,}")

    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    tests = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        X, _, _ = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                  with_target=False)
        tests.append(X[:, keep])

    ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                     categorical_feature=cats, free_raw_data=False)
    for (label, over, rounds), r in zip(CANDS, res):
        t1 = time.time()
        p = dict(COMMON, **over)
        models = [lgb.train(dict(p, seed=sd), ds, num_boost_round=rounds)
                  for sd in SUBMIT_SEEDS]
        out = tpl.copy()
        out[ctx.items] = out[ctx.items].astype(float)
        for t, X in enumerate(tests):
            pr = np.mean([np.expm1(mdl.predict(X)) for mdl in models], 0)
            pr = np.maximum(pr, FLOOR).reshape(C.HORIZON, ctx.n)   # 평균 뒤 하한 1회
            for h in range(1, C.HORIZON + 1):
                out.loc[out.index[out[rk] == f"TEST_{t:02d}+{h}일"][0],
                        ctx.items] = pr[h - 1]
        out[ctx.items] = out[ctx.items].round(2)
        stamp = "v7_ff65_l255" if label.startswith("A") else "v8_ff65_l127"
        path = os.path.join(C.SUBMISSIONS, f"submission_{stamp}.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        v = out[ctx.items].values.astype(float)
        assert not np.isnan(v).any() and (v >= FLOOR - 1e-9).all()
        r["file"] = f"submission_{stamp}.csv"
        print(f"\n  저장: submission_{stamp}.csv   ({label}, {time.time()-t1:.0f}s)")
        print(f"    min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
              f"평균 {v.mean():.2f} / max {v.max():.2f}")
        q = os.path.join(C.SUBMISSIONS, "submission_v3_xgb.csv")
        if os.path.exists(q):
            o = pd.read_csv(q)[ctx.items].values.astype(float)
            print(f"    현재 최고(v3 XGB) 대비 로그공간 상관 "
                  f"{np.corrcoef(np.log1p(o).ravel(), np.log1p(v).ravel())[0,1]:.4f}")

    json.dump(dict(seeds=list(SEEDS), ref_v1=REF_V1, results=res),
              open(os.path.join(C.EXPERIMENTS, "phase5g_colsample.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: experiments/phase5g_colsample.json")
    print(f"총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
