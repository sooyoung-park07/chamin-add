# -*- coding: utf-8 -*-
"""Phase 4-c — 검증 정교화(origin 7일→1일) 후 노이즈 재측정 및 기존 결론 재판정.

핵심: 학습은 그대로, 채점만 촘촘하게. 노이즈가 얼마나 줄었는지 먼저 재고,
그 새 기준으로 지금까지의 애매한 결론들을 다시 판정한다.
"""
import os
import json
import time

import numpy as np
import lightgbm as lgb

import config as C
import dataio as D
import features as F
import validate as V
from metrics import competition_score, make_weights

SCREEN = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=os.cpu_count())
ROUNDS, FLOOR = 600, 1.0
DOMAIN_FLAGS = {"hwadam_open", "ski_season", "ski_peak", "foliage"}
RESULTS = []


def run(tag, folds, keep_names, seeds=(42, 7, 2024), record=True):
    fn = F.feature_names()
    keep = [i for i, k in enumerate(fn) if k in keep_names]
    names = [fn[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]
    scores, t0 = [], time.time()
    for fd in folds:
        m = fd["ytr"] > 0
        preds = []
        for sd in seeds:
            ds = lgb.Dataset(fd["Xtr"][m][:, keep], label=np.log1p(fd["ytr"][m]),
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            mdl = lgb.train(dict(SCREEN, seed=sd), ds, num_boost_round=ROUNDS)
            preds.append(np.expm1(mdl.predict(fd["Xva"][:, keep])))
        p = np.maximum(np.mean(preds, 0), FLOOR)
        scores.append(competition_score(fd["yva"], p, fd["iids"],
                                        fd["store_of"], make_weights(1.0), fd["n"]))
    cv3, cv23 = float(np.mean(scores)), float(np.mean(scores[1:]))
    if record:
        RESULTS.append(dict(tag=tag, cv3=cv3, cv23=cv23, n_feat=len(names),
                            folds=[float(s) for s in scores]))
        print(f"  {tag:<30s} cv3 {cv3:.4f} | F2+F3 {cv23:.4f}  "
              f"({' '.join(f'{s:.4f}' for s in scores)})  [{time.time()-t0:.0f}s]")
    return cv3, cv23


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    ALL = set(F.feature_names())
    G = F.FEATURE_GROUPS

    print("=" * 82)
    print("검증 정교화 — origin 간격 7일 → 1일")
    print("=" * 82)
    folds, t0 = [], time.time()
    for name, d0, d1 in V.FOLDS:
        va = V.origins(dates, d0, d1, nd)               # 이제 step=1이 기본
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        folds.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                          iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))
        print(f"  {name}: 학습 origin {len(trn)} · 검증 origin {len(va)} "
              f"· 채점행 {int((yva != 0).sum()):,}")
    print(f"  (행렬 생성 {time.time()-t0:.0f}s)\n")

    print("=" * 82)
    print("① 새 노이즈 바닥 — 같은 설정, 시드만 다르게 (시드쌍 5개)")
    print("=" * 82)
    n3, n23 = [], []
    for sds in [(42, 7), (11, 13), (101, 103), (999, 1001), (5, 55)]:
        a, b = run("", folds, ALL, seeds=sds, record=False)
        n3.append(a)
        n23.append(b)
        print(f"  시드 {str(sds):<14s} cv3 {a:.4f} | F2+F3 {b:.4f}")
    s3, s23 = float(np.std(n3)), float(np.std(n23))
    print(f"\n  cv3    평균 {np.mean(n3):.4f} · σ {s3:.4f} · 2σ {2*s3:.4f}")
    print(f"  F2+F3  평균 {np.mean(n23):.4f} · σ {s23:.4f} · 2σ {2*s23:.4f}")
    print(f"\n  [이전 기준] step=7일 때 cv3 σ 0.0014 · 2σ 0.0028")
    print(f"  [현재 기준] step=1일 때 cv3 σ {s3:.4f} · 2σ {2*s3:.4f}"
          f"   → 노이즈 {0.0014/max(s3,1e-9):.1f}배 감소")

    print()
    print("=" * 82)
    print("② 새 자로 기존 결론 재판정 (판정은 F2+F3 기준)")
    print("=" * 82)
    base3, base23 = run("BASE (현재 구성)", folds, ALL)
    cands = [
        ("− closed 그룹", ALL - set(G["closed"]), "Phase2에서 채택했던 것"),
        ("− 도메인 플래그", ALL - DOMAIN_FLAGS, "내가 '잉여'라 했다 철회한 것"),
        ("− item_id", ALL - {"item_id"}, "cv3와 F2+F3가 갈렸던 것"),
        ("− dow 그룹", ALL - set(G["dow"]), "'중복'이라 했던 것"),
        ("− ctx 그룹", ALL - set(G["ctx"]), "'기여 미미'라 했던 것"),
        ("− 공휴일", ALL - {"is_holiday", "is_holiday_eve"}, ""),
    ]
    for tag, keep, note in cands:
        run(tag, folds, keep)

    print()
    print("=" * 82)
    print(f"판정  (BASE F2+F3 {base23:.4f} 대비 · 2σ={2*s23:.4f})")
    print("=" * 82)
    for r in sorted(RESULTS, key=lambda x: x["cv23"]):
        d = r["cv23"] - base23
        if abs(d) < 2 * s23:
            v = "차이 없음"
        elif d < 0:
            v = "★ 빼는 게 낫다"
        else:
            v = "유지해야 한다"
        print(f"  {r['tag']:<30s} F2+F3 {r['cv23']:.4f}  {d:+.4f}   {v}")

    json.dump({"noise_cv3_std": s3, "noise_cv23_std": s23,
               "noise_runs_cv3": n3, "noise_runs_cv23": n23, "results": RESULTS},
              open(os.path.join(C.EXPERIMENTS, "phase4c_precision.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4c_precision.json")


if __name__ == "__main__":
    main()
