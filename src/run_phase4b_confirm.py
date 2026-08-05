# -*- coding: utf-8 -*-
"""Phase 4-b — 노이즈 바닥 측정 + ablation 유망 후보 검증.

왜 필요한가:
  4-a에서 관측된 차이 상당수가 0.003 수준이다. **노이즈 바닥을 모르면
  0.003이 실제 개선인지 우연인지 구분할 수 없다.** 같은 설정을 시드만 바꿔
  여러 번 돌려 표준편차를 먼저 재고, 그보다 큰 차이만 신뢰한다.
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


def run(tag, folds, keep_names, seeds=(42, 7, 2024), quiet=False):
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
    m3 = float(np.mean(scores))
    if not quiet:
        RESULTS.append(dict(tag=tag, cv3=m3, cv23=float(np.mean(scores[1:])),
                            n_feat=len(names)))
        print(f"  {tag:<32s} {m3:.4f}  (피처 {len(names):>2d}개, "
              f"{' '.join(f'{s:.4f}' for s in scores)})  [{time.time()-t0:.0f}s]")
    return m3


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    fn = F.feature_names()
    ALL = set(fn)
    G = F.FEATURE_GROUPS

    folds = []
    for name, d0, d1 in V.FOLDS:
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        folds.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                          iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))

    print("=" * 80)
    print("① 노이즈 바닥 — 같은 설정, 시드만 다르게")
    print("=" * 80)
    noise = []
    for sds in [(42, 7), (11, 13), (101, 103), (999, 1001), (5, 55)]:
        s = run("", folds, ALL, seeds=sds, quiet=True)
        noise.append(s)
        print(f"  시드 {str(sds):<14s} {s:.4f}")
    sd = float(np.std(noise))
    print(f"\n  평균 {np.mean(noise):.4f} · 표준편차 {sd:.4f} · "
          f"범위 {min(noise):.4f}~{max(noise):.4f}")
    print(f"  → **차이가 {2*sd:.4f} (2σ) 미만이면 우연일 수 있다.** 이 기준으로 아래를 판독한다.")

    print()
    print("=" * 80)
    print("② 유망 후보 검증 (시드 3개)")
    print("=" * 80)
    base = run("BASE (price 숫자 복구)", folds, ALL)
    cands = [
        ("− item_id", ALL - {"item_id"}),
        ("− 도메인 플래그", ALL - DOMAIN_FLAGS),
        ("− item_id − 도메인", ALL - {"item_id"} - DOMAIN_FLAGS),
        ("− dow 그룹", ALL - set(G["dow"])),
        ("− ctx 그룹", ALL - set(G["ctx"])),
        ("− item_id − 도메인 − ctx", ALL - {"item_id"} - DOMAIN_FLAGS - set(G["ctx"])),
        ("− 위 전부 − dow", ALL - {"item_id"} - DOMAIN_FLAGS - set(G["ctx"]) - set(G["dow"])),
    ]
    for tag, keep in cands:
        run(tag, folds, keep)

    print()
    print("=" * 80)
    print(f"판정 (BASE {base:.4f} 대비, 2σ={2*sd:.4f})")
    print("=" * 80)
    for r in sorted(RESULTS, key=lambda x: x["cv3"]):
        d = r["cv3"] - base
        if abs(d) < 2 * sd:
            verdict = "차이 없음(노이즈 범위)"
        elif d < 0:
            verdict = "★ 개선"
        else:
            verdict = "악화"
        print(f"  {r['tag']:<32s} {r['cv3']:.4f}  {d:+.4f}   {verdict}")

    json.dump({"noise_std": sd, "noise_runs": noise, "results": RESULTS},
              open(os.path.join(C.EXPERIMENTS, "phase4b_confirm.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4b_confirm.json")


if __name__ == "__main__":
    main()
