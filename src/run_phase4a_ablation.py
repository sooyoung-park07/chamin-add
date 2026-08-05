# -*- coding: utf-8 -*-
"""Phase 4-a — 피처 그룹 ablation.

목적은 점수 개선이 아니라 **"이 문제가 무엇으로 풀리는가"를 표로 만드는 것**.

두 방향으로 본다 (하나만 보면 오해한다):
  ① Leave-one-out (빼기) : 그 그룹만 제거 → **고유 기여도**.
     다른 그룹이 대체할 수 있으면 빼도 점수가 안 떨어진다(= 중복).
  ② Only-one (홀로)      : 그 그룹만 사용 → **단독 능력**.
     혼자서도 강한지. ①이 작아도 ②가 크면 "중요하지만 대체 가능"이란 뜻.
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
ROUNDS, SEEDS, FLOOR = 600, (42, 7), 1.0
RESULTS = []


def run(tag, folds, keep_names, note=""):
    fn = F.feature_names()
    keep = [i for i, k in enumerate(fn) if k in keep_names]
    names = [fn[i] for i in keep]
    cats = [c for c in F.CATEGORICAL if c in names]
    scores, t0 = [], time.time()
    for fd in folds:
        m = fd["ytr"] > 0
        preds = []
        for sd in SEEDS:
            ds = lgb.Dataset(fd["Xtr"][m][:, keep], label=np.log1p(fd["ytr"][m]),
                             feature_name=names, categorical_feature=cats,
                             free_raw_data=False)
            mdl = lgb.train(dict(SCREEN, seed=sd), ds, num_boost_round=ROUNDS)
            preds.append(np.expm1(mdl.predict(fd["Xva"][:, keep])))
        p = np.maximum(np.mean(preds, 0), FLOOR)
        scores.append(competition_score(fd["yva"], p, fd["iids"],
                                        fd["store_of"], make_weights(1.0), fd["n"]))
    m3 = float(np.mean(scores))
    RESULTS.append(dict(tag=tag, cv3=m3, cv23=float(np.mean(scores[1:])),
                        n_feat=len(names), note=note))
    print(f"  {tag:<30s} {m3:.4f}  (피처 {len(names):>2d}개, "
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

    print("피처 그룹 구성")
    for g, keys in G.items():
        print(f"  {g:<8s} {len(keys):>2d}개  {', '.join(keys[:4])}"
              f"{' ...' if len(keys) > 4 else ''}")
    print()

    folds = []
    for name, d0, d1 in V.FOLDS:
        va = V.origins(dates, d0, d1, nd)
        trn = V.train_origins(dates, d0, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        folds.append(dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                          iids=mva[:, 2], store_of=ctx.store_of_item, n=ctx.n))

    print("=" * 80)
    print("기준 — 전체 피처")
    print("=" * 80)
    base = run("FULL (전체)", folds, ALL)

    print()
    print("=" * 80)
    print("① Leave-one-out — 이 그룹을 빼면 얼마나 나빠지나 (고유 기여도)")
    print("=" * 80)
    loo = {}
    for g, keys in G.items():
        loo[g] = run(f"− {g}", folds, ALL - set(keys)) - base

    print()
    print("=" * 80)
    print("② Only-one — 이 그룹만 쓰면 얼마나 되나 (단독 능력)")
    print("=" * 80)
    only = {}
    for g, keys in G.items():
        only[g] = run(f"only {g}", folds, set(keys))

    print()
    print("=" * 80)
    print("③ 핵심 개별 피처")
    print("=" * 80)
    extras = [
        ("− item_id만", ALL - {"item_id"}, "품목 식별자만 제거 (속성은 유지)"),
        ("− 요일정보 전부", ALL - set(G["dow"]) - {"dow", "is_weekend", "is_dayoff"},
         "요일 관련 싹 제거"),
        ("− 최근수준 3종", ALL - {"w_posmean", "w_mean_open", "d_posmean"},
         "창의 '평균적으로 얼마 팔리나' 제거"),
        ("− 도메인 플래그", ALL - {"hwadam_open", "ski_season", "ski_peak", "foliage"},
         "리조트 도메인 지식 제거"),
        ("− 공휴일", ALL - {"is_holiday", "is_holiday_eve"}, "공휴일 정보 제거"),
    ]
    for tag, keep, note in extras:
        run(tag, folds, keep, note)

    # ---------------- 중요도 (gain) ----------------
    print()
    print("=" * 80)
    print("④ 학습된 모델의 gain 중요도 (마지막 폴드)")
    print("=" * 80)
    fd = folds[-1]
    m = fd["ytr"] > 0
    ds = lgb.Dataset(fd["Xtr"][m], label=np.log1p(fd["ytr"][m]), feature_name=fn,
                     categorical_feature=F.CATEGORICAL, free_raw_data=False)
    mdl = lgb.train(dict(SCREEN, seed=42), ds, num_boost_round=ROUNDS)
    gain = mdl.feature_importance("gain")
    tot = gain.sum()
    bygroup = {g: sum(gain[fn.index(k)] for k in keys) for g, keys in G.items()}
    print("  [그룹별 gain 비중]")
    for g, v in sorted(bygroup.items(), key=lambda x: -x[1]):
        print(f"    {g:<8s} {100*v/tot:>5.1f}%")
    print("\n  [개별 상위 15]")
    for k, v in sorted(zip(fn, gain), key=lambda x: -x[1])[:15]:
        grp = next(g for g, ks in G.items() if k in ks)
        print(f"    {k:<18s} {100*v/tot:>5.1f}%   ({grp})")

    # ---------------- 요약 ----------------
    print()
    print("=" * 80)
    print("요약 — 그룹별 고유기여 vs 단독능력  (기준 FULL = %.4f)" % base)
    print("=" * 80)
    print(f"  {'그룹':<8s} {'빼면 악화':>10s} {'홀로 쓰면':>10s} {'gain비중':>9s}   해석")
    for g in sorted(G, key=lambda x: -loo[x]):
        d, o, gp = loo[g], only[g], 100 * bygroup[g] / tot
        if d >= 0.010:
            interp = "대체 불가 — 핵심"
        elif d >= 0.003:
            interp = "고유 기여 있음"
        elif o < base + 0.08:
            interp = "강하지만 다른 그룹과 중복"
        else:
            interp = "기여 작음"
        print(f"  {g:<8s} {d:>+10.4f} {o:>10.4f} {gp:>8.1f}%   {interp}")

    json.dump(RESULTS, open(os.path.join(C.EXPERIMENTS, "phase4a_ablation.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase4a_ablation.json")


if __name__ == "__main__":
    main()
