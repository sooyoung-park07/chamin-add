# -*- coding: utf-8 -*-
"""Phase 13 — `n_od` 피처 실험 (Q5). 제출 없음, 내부 4폴드 × 짝시드.

n_od = 창 안 '대상일과 같은 요일' 4일 중 그 영업장의 **영업일 수**.
    화담숲처럼 46%가 휴무인 곳은 4일 중 1~2일만 영업한 창이 흔한데,
    지금 피처(x_store_dow)는 휴점일을 0으로 포함해 평균 내므로 모델이 '수요 감소'로 오해한다.
    오라클 배율 1.92/1.76/1.34/1.18/0.98 (n_od=0..4, 완전 단조) · TEST 노출 28.6% · 천장 +0.0065.

사전등록 (실행 전 고정)
    변형 N0: 현행 57개 (대조군, 같은 시드)
    변형 N1: 57 + d_nopen(품목 영업장의 n_od, 0~4 정수) = 58개
    변형 N2: N1 에서 x_store_dow 를 영업일 평균으로 교체 (n_od=0 이면 **NaN** — LightGBM 네이티브 결측.
             사후에 다른 처리와 비교하지 않는다. 자유도 차단.)
    판정: N1·N2 각각 N0 대비, 같은 시드 8개 짝비교 · 4폴드.
          채택 = 4/4 같은 방향 AND 평균 개선 > 0.001. 최댓값 선택 금지 — N1/N2 중
          더 좋은 쪽을 고르는 게 아니라, 둘 다 방향이 맞을 때만 단순한 쪽(N1) 우선.
    채점: v17 후처리(배율→하한→기하스냅) 통과 후. 램프부스트 적용판도 병기
          (전환기 신호가 겹치므로 — 부스트가 이미 먹은 걸 피처가 또 먹는지 확인).
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

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]


def seg_base(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
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


def extra_cols(mat, dates, meta, store_codes, ctx):
    """d_nopen · x_store_dow_open 을 build_samples 밖에서 계산."""
    didx = pd.DatetimeIndex(dates)
    wd_all = didx.dayofweek.values
    ns = store_codes.max() + 1
    st_daily = np.zeros((ns, mat.shape[1]))
    for s in range(ns):
        st_daily[s] = np.nansum(np.abs(mat[store_codes == s]), 0)
    open_ = st_daily > 0
    st_signed = np.zeros((ns, mat.shape[1]))
    for s in range(ns):
        st_signed[s] = np.nansum(mat[store_codes == s], 0)

    n = len(meta)
    d_nopen = np.empty(n); x_open = np.empty(n)
    cache = {}
    for i in range(n):
        o, h, it = meta[i]
        key = (o, h)
        if key not in cache:
            wd = wd_all[o + h]
            days = [j for j in range(o - 27, o + 1) if wd_all[j] == wd]
            days = np.array(days)
            nop = open_[:, days].sum(1).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                so = np.where(nop > 0,
                              np.array([st_signed[s, days][open_[s, days]].sum()
                                        for s in range(ns)]) / np.maximum(nop, 1),
                              np.nan)
            cache[key] = (nop, so)
        nop, so = cache[key]
        s = store_codes[it]
        d_nopen[i] = nop[s]
        x_open[i] = so[s]
    return d_nopen, x_open


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]
    ix_xsd = names.index("x_store_dow")

    print("=" * 96)
    print(f"Phase 13 — n_od 실험 · 시드 {len(SEEDS)}개 짝비교 · 3변형 × 4폴드")
    print("=" * 96, flush=True)

    results = {}
    for fi, (fname, cut, v0, v1) in enumerate(FOLDS):
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        tr_extra = extra_cols(mat, dates, mtr[m], ctx.store_codes, ctx)
        va_extra = extra_cols(mat, dates, mva, ctx.store_codes, ctx)
        Xt = np.ascontiguousarray(Xtr[m][:, keep]); yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv = np.ascontiguousarray(Xva[:, keep])
        del Xtr, Xva
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        # 램프 마스크 (겹침 확인용)
        r = np.empty(len(yva)); h7 = np.zeros(len(yva), dtype=int)
        hols = set(pd.to_datetime(list(C.HOLIDAYS)))
        hol_arr = np.array([d in hols for d in pd.DatetimeIndex(dates)])
        for o in np.unique(mva[:, 0]):
            mm = mva[:, 0] == o
            it = mva[mm, 2]
            l7 = mat[:, o - 6:o + 1].mean(1); p7 = mat[:, o - 13:o - 6].mean(1)
            r[mm] = (l7[it] + 1.0) / (p7[it] + 1.0)
            h7[mm] = int(hol_arr[o - 6:o + 1].sum())
        ramp = (r > 2.0) & (h7 < 2)

        def make_xy(variant):
            if variant == "N0":
                return Xt, Xv, list(names)
            if variant == "N1":
                a = np.column_stack([Xt, tr_extra[0]])
                b = np.column_stack([Xv, va_extra[0]])
                return a, b, names + ["d_nopen"]
            a = np.column_stack([Xt, tr_extra[0]]); a[:, ix_xsd] = tr_extra[1]
            b = np.column_stack([Xv, va_extra[0]]); b[:, ix_xsd] = va_extra[1]
            return a, b, [x if x != "x_store_dow" else "x_store_dow_open"
                          for x in names] + ["d_nopen"]

        for variant in ("N0", "N1", "N2"):
            t1 = time.time()
            Xa, Xb, nm = make_xy(variant)
            preds = []
            for sd in SEEDS:
                ds = lgb.Dataset(np.ascontiguousarray(Xa), label=yt, feature_name=nm,
                                 categorical_feature=cats, free_raw_data=False)
                preds.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                                num_boost_round=ROUNDS).predict(Xb)))
                del ds
            raw = np.mean(preds, 0)
            base = seg_base(raw)
            sc_plain = float((W * loss(yva, gsnap(base))).sum())
            pb = base.copy(); pb[ramp] = 1.5 * pb[ramp]
            sc_ramp = float((W * loss(yva, gsnap(pb))).sum())
            results[(fname, variant)] = (sc_plain, sc_ramp)
            print(f"  [{fname:<8s}][{variant}] 후처리만 {sc_plain:.5f} · +램프 {sc_ramp:.5f}"
                  f"   ({time.time()-t1:.0f}s)", flush=True)

    print("\n" + "=" * 96)
    print("결과 — N0 대비 개선 (양수 = 좋음)")
    print("=" * 96)
    for tag, j in (("후처리만", 0), ("+램프부스트", 1)):
        print(f"\n  [{tag}]")
        print(f"  {'변형':<6s}" + "".join(f"{f[0]:>11s}" for f in FOLDS) + f"{'평균':>10s}{'일관':>6s}")
        for variant in ("N1", "N2"):
            g = [results[(f[0], "N0")][j] - results[(f[0], variant)][j] for f in FOLDS]
            print(f"  {variant:<6s}" + "".join(f"{x:>+11.5f}" for x in g)
                  + f"{np.mean(g):>+10.5f}{sum(x > 0 for x in g):>4d}/4")

    json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()},
              open(os.path.join(C.EXPERIMENTS, "phase13_nod.json"), "w"), indent=1)
    print(f"\n총 {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
