# -*- coding: utf-8 -*-
"""Phase 4-d — 노이즈를 실제로 줄이는 법 찾기.

4-c에서 '검증 표본을 7배로 늘려도 노이즈가 안 줄었다'는 걸 확인했다.
노이즈의 출처가 표본이 아니라 **모델(시드)**이기 때문. 그렇다면 두 가지가 남는다.

  ① 시드 수를 늘린다 → σ가 1/√n 로 줄어드는가?
  ② A/B를 **같은 시드로 짝지어** 비교한다 → 차이(A−B)의 σ가 개별 σ보다 작은가?
     (공통 성분이 상쇄되면 훨씬 적은 비용으로 같은 판별력을 얻는다)

구현 요령: 설정마다 시드별 예측을 **한 번만 계산해 캐시**해두면,
어떤 시드 조합의 평균이든 재학습 없이 즉시 점수를 낼 수 있다.
"""
import os
import json
import time
import itertools

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
SEED_POOL = [42, 7, 2024, 913, 31, 11, 13, 101, 103, 999,
             1001, 5, 55, 201, 203, 301, 303, 401, 403, 501]


def train_cache(folds, keep_names, seeds):
    """설정 하나에 대해 시드별 검증 예측을 계산해 캐시."""
    fn = F.feature_names()
    keep = [i for i, k in enumerate(fn) if k in keep_names]
    names = [fn[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]
    cache = []                       # cache[fold][seed] = 예측 배열
    for fd in folds:
        m = fd["ytr"] > 0
        Xt, yt = fd["Xtr"][m][:, keep], np.log1p(fd["ytr"][m])
        Xv = fd["Xva"][:, keep]
        per = {}
        for sd in seeds:
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            mdl = lgb.train(dict(SCREEN, seed=sd), ds, num_boost_round=ROUNDS)
            per[sd] = np.expm1(mdl.predict(Xv))
        cache.append(per)
    return cache


def score_subset(folds, cache, seeds):
    """캐시된 예측에서 특정 시드 조합의 앙상블 점수(재학습 없음)."""
    out = []
    for fd, per in zip(folds, cache):
        p = np.maximum(np.mean([per[s] for s in seeds], axis=0), FLOOR)
        out.append(competition_score(fd["yva"], p, fd["iids"],
                                     fd["store_of"], make_weights(1.0), fd["n"]))
    return float(np.mean(out)), float(np.mean(out[1:]))


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    ALL = set(F.feature_names())

    folds = []
    for name, d0, d1 in V.FOLDS:
        va = V.origins(dates, d0, d1, nd)          # step=7로 복귀
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        folds.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                          iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))
    print(f"검증 origin: {[len(V.origins(dates, d0, d1, nd)) for _, d0, d1 in V.FOLDS]}"
          f" (step=7로 복귀)\n")

    t0 = time.time()
    print(f"BASE 설정 학습 중 (시드 {len(SEED_POOL)}개 × 3폴드)...")
    base_cache = train_cache(folds, ALL, SEED_POOL)
    print(f"  완료 [{time.time()-t0:.0f}s]\n")

    print("=" * 78)
    print("① 시드 수 n 에 따른 노이즈 σ  — 1/√n 로 줄어드는가?")
    print("=" * 78)
    rng = np.random.default_rng(0)
    print(f"  {'n':>3s} {'복제수':>5s} {'평균':>9s} {'σ(cv3)':>9s} {'σ(F2+F3)':>10s} {'예측 1/√n':>10s}")
    sig1 = None
    for n in (1, 2, 4, 8):
        reps = []
        pool = list(SEED_POOL)
        rng.shuffle(pool)
        # 겹치지 않는 조합으로 복제 만들기
        k = len(pool) // n
        for i in range(min(k, 8)):
            reps.append(score_subset(folds, base_cache, pool[i * n:(i + 1) * n]))
        c3 = [r[0] for r in reps]
        c23 = [r[1] for r in reps]
        s3 = float(np.std(c3))
        if n == 1:
            sig1 = s3
        pred = sig1 / np.sqrt(n)
        print(f"  {n:>3d} {len(reps):>5d} {np.mean(c3):>9.4f} {s3:>9.4f} "
              f"{np.std(c23):>10.4f} {pred:>10.4f}")

    print()
    print("=" * 78)
    print("② 같은 시드로 짝지어 비교하면(paired) 정밀도가 오르는가?")
    print("=" * 78)
    print("  비교 대상: BASE vs '− closed 그룹' (4-c에서 +0.0036, 실재하는 효과)")
    var_cache = train_cache(folds, ALL - set(F.FEATURE_GROUPS["closed"]),
                            SEED_POOL[:10])
    pairs = [SEED_POOL[i:i + 2] for i in range(0, 10, 2)]
    diffs3, diffs23, a_list, b_list = [], [], [], []
    for sds in pairs:
        a3, a23 = score_subset(folds, base_cache, sds)
        b3, b23 = score_subset(folds, var_cache, sds)
        a_list.append(a23)
        b_list.append(b23)
        diffs3.append(b3 - a3)
        diffs23.append(b23 - a23)
        print(f"  시드 {str(sds):<12s} BASE {a23:.4f} · 제거 {b23:.4f} · 차이 {b23-a23:+.4f}")
    print(f"\n  개별 점수의 σ      : BASE {np.std(a_list):.4f} · 제거 {np.std(b_list):.4f}")
    print(f"  **차이(paired)의 σ** : {np.std(diffs23):.4f}   "
          f"(평균 차이 {np.mean(diffs23):+.4f})")
    ratio = np.std(diffs23) / max(np.std(a_list), 1e-9)
    print(f"  → 짝지어 비교하면 노이즈가 개별 대비 {ratio:.2f}배")
    if ratio < 0.8:
        print("     **짝비교가 유효하다. 같은 시드를 고정해 A/B하면 훨씬 적은 비용으로 판정 가능.**")
    else:
        print("     짝비교 이득이 작다 → 시드 수를 늘리는 수밖에 없다.")

    json.dump({"note": "phase4d", "seed_pool": SEED_POOL,
               "paired_diff_std": float(np.std(diffs23)),
               "paired_diff_mean": float(np.mean(diffs23))},
              open(os.path.join(C.EXPERIMENTS, "phase4d_noise.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4d_noise.json")


if __name__ == "__main__":
    main()
