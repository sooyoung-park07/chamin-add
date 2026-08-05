# -*- coding: utf-8 -*-
"""Phase 16 — E2: 창 상대 타깃 (드리프트 강건 설계).

동기 (점수 없이 성립하는 근거 사슬 — 이게 이 실험의 존재 이유다):
    ① 창 YoY 측정(각 TEST 창 ÷ 학습연도 같은 기간, 합법): 테스트 연도가 0.75~0.92 로 약하다
    ② 탄력성 측정(내부 반사실): 현행 모델은 창 변화의 31~49% 만 따라간다
    → 타깃을 '창 대비 몇 배'로 바꾸면 구조적으로 연도 수준을 따라간다 (탄력성 ≈ 1)

설계:
    현행 N0 :  f(x) ≈ log1p(y)                      → p = expm1(f)
    E2      :  f(x) ≈ log1p(y) − log1p(w_posmean)   → p = expm1(f + log1p(w_posmean))
    피처 57개 동일 · 같은 시드 8개 짝비교 · v17 후처리 동일 → 바뀌는 것은 타깃 변환 하나.
    anchor = w_posmean (창 안 '팔린 날'들의 평균 = "팔린다면 얼마"의 창 기준점).
    anchor=0 (창 전체 무판매) 이면 절대 타깃으로 자연 폴백.

사전등록 (실행 전 고정):
    · 내부 판정: 내부는 검증연도=학습연도라 **드리프트 이득을 원리적으로 볼 수 없다.**
      따라서 채택 기준은 "이겨라"가 아니라 **"본전 근처"**: 4폴드 평균 손실 < 0.002
      AND 최악 폴드 손실 < 0.005 → 확인 제출 1회 진행.
    · 실전 비교 대상: v20 (같은 파이프라인·같은 시드5·LGBM 단독·같은 후처리) Private 0.4390529.
    · 실전 기대: −0.003 ~ +0.003 (창 추종이 모든 구간에서 유리하다는 보장은 없으므로 부호 미정).
    · 부수 검증: E2 의 탄력성이 실제로 ≈1 인지 반사실로 확인 (기전 확인).
    · 후처리 정합: raw<1.818 비율이 N0 대비 3%p 초과 이탈하면 판정 보류 (진단 규칙).
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
from run_phase10c_thresholds import cell_weights

SEEDS = (42, 7, 2024, 913, 31, 101, 202, 303)
NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]


def seg_base(r):
    p = np.where(r < 1.8, 0.55 * r, np.where(r < 10.0, 0.90 * r, 1.02 * r))
    return np.maximum(p, 1.0)


def gsnap(p):
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a = np.abs(a); p = np.abs(p)
    den = a + p
    out = np.zeros(len(a)); m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]
    ia = names.index("w_posmean")

    print("=" * 96)
    print(f"Phase 16 — E2 창 상대 타깃 vs N0 · 시드 {len(SEEDS)}개 짝비교")
    print("=" * 96, flush=True)

    results = {}
    e2_models_f2 = []
    for fi, (fname, cut, v0, v1) in enumerate(FOLDS):
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        anc_t = np.log1p(np.maximum(Xt[:, ia], 0.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        anc_v = np.log1p(np.maximum(Xv[:, ia], 0.0))
        del Xtr, Xva
        yt_abs = np.log1p(np.maximum(ytr[m], 1.0))
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)

        for variant in ("N0", "E2"):
            t1 = time.time()
            label = yt_abs if variant == "N0" else yt_abs - anc_t
            preds = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=label, feature_name=names,
                                 categorical_feature=cats, free_raw_data=False)
                md = lgb.train(dict(PARAMS, seed=sd), ds, num_boost_round=ROUNDS)
                f_out = md.predict(Xv)
                preds.append(np.maximum(
                    np.expm1(f_out if variant == "N0" else f_out + anc_v), 0.0))
                if variant == "E2" and fname == "F2 겨울" and len(e2_models_f2) < 3:
                    e2_models_f2.append(md)
                del ds
            raw = np.mean(preds, 0)
            sc = float((W * loss(yva, gsnap(seg_base(raw)))).sum())
            frac = float((raw < 1.818).mean())
            results[(fname, variant)] = (sc, frac)
            print(f"  [{fname:<8s}][{variant}] {sc:.5f} · raw<1.818 {100*frac:.1f}%"
                  f"   ({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 96)
    print("결과 — E2 의 N0 대비 (양수 = E2 가 좋음)")
    print("=" * 96)
    g = [results[(f[0], "N0")][0] - results[(f[0], "E2")][0] for f in FOLDS]
    print("  " + " ".join(f"{f[0]} {x:+.5f}" for f, x in zip(FOLDS, g))
          + f"   평균 {np.mean(g):+.5f} · 일관 {sum(x > 0 for x in g)}/4")
    print("  raw<1.818 차이: " + " ".join(
        f"{100*(results[(f[0],'E2')][1]-results[(f[0],'N0')][1]):+.1f}%p" for f in FOLDS))
    ok = (-np.mean(g) < 0.002) and (max(-x for x in g) < 0.005)
    print(f"\n  사전등록 판정 (평균손실<0.002 AND 최악<0.005): {'통과 → 확인 제출 진행' if ok else '탈락 → 종료'}")

    # ── 부수 검증: E2 탄력성 (F2 · 3시드 · c=0.7) ─────────────────────
    print("\n" + "=" * 96)
    print("부수 검증 — E2 탄력성 (기대 ≈ 1)")
    print("=" * 96, flush=True)
    fname, cut, v0, v1 = FOLDS[0]
    cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
    ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
    va = V.origins(dates, v0, v1, nd)
    ps = {}
    for c in (1.0, 0.7):
        mm = mat if c == 1.0 else mat * c
        Xva, yva, mva = F.build_samples(mm, dates, va, ctx)
        Xv = np.ascontiguousarray(Xva[:, keep])
        anc = np.log1p(np.maximum(Xv[:, ia], 0.0))
        ps[c] = np.mean([np.expm1(md.predict(Xv) + anc) for md in e2_models_f2], 0)
        del Xva, Xv
    W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
    p0 = np.maximum(ps[1.0], 0.0)
    okm = valid & (p0 > 0.3)
    e = float(np.average(((np.log1p(np.maximum(ps[0.7], 0.0)) - np.log1p(p0))
                          / np.log(0.7))[okm], weights=W[okm]))
    print(f"  E2 전체 탄력성 (c=0.7): {e:.3f}   (현행 N0 은 0.49 였다)")

    json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()},
              open(os.path.join(C.EXPERIMENTS, "phase16_relative.json"), "w"), indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
