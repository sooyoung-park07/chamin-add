# -*- coding: utf-8 -*-
"""Tier B — TB7 부속 2: 봄 이득의 origin 분해 (규칙 ㉒ 검사).

배경: v27 계절 게이트의 패인 = 봄 이득 +0.00330의 90%가 origin 1개(2024-03-29 화담숲
개장주간)에 몰려 있었음(규칙 ㉒·⑦). TB7b의 봄 이득(F3 +0.00928 / FAR-봄 +0.01209)이
같은 함정인지, 여러 주에 고루 퍼진 진짜 신호인지 origin별로 분해한다.

판단 기준(사전 기록): top-1 origin 기여가 전체의 50%를 넘으면 v27형(불신·라우팅 반대),
전 origin의 60% 이상이 양수이고 top-1이 40% 미만이면 퍼진 신호(라우팅/전면 적용 검토 가치).
저울·시드는 TB7과 동일 — base(57) 재현값이 TB7과 일치하는지로 자가검증.
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

NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS, SEEDS = 1000, (42, 7, 2024, 913, 31)
DROP3 = ["w_posmedian", "w_last14", "w_std"]

FOLDS = [("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08")]
OUT = os.path.join(C.EXPERIMENTS, "phase_tb7_spring_decomp.json")


def post(raw, seg=True, snap=True):
    p = (np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
         if seg else raw.copy())
    p = np.maximum(p, 1.0)
    if not snap:
        return p
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


# ---- TB7과 동일한 베타 파이프라인 (run_tb7_visit_weather.py에서 그대로) ----
def load_weather_daily():
    df = pd.read_csv(os.path.join(C.DATA, "tierb", "weather_icheon.csv"),
                     encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ta = df["ta_avg"].astype(float)
    anom = (ta - ta.rolling(15, center=True, min_periods=1).mean()).to_numpy()
    rain = (df["rn_day"].astype(float).to_numpy() >= 1.0).astype(float)
    return (dict(zip(df["date"], anom)), dict(zip(df["date"], rain)))


def day_arrays(dates, anom, rain):
    a = np.zeros(len(dates), np.float32)
    r = np.zeros(len(dates), np.float32)
    for j, dt in enumerate(dates):
        if dt in anom:
            a[j] = anom[dt]
            r[j] = rain[dt]
    return a, r


def item_betas(mat, dates, cut_col, anom_by_day, rain_by_day):
    n, _ = mat.shape
    beta_ta = np.zeros(n, np.float32)
    beta_rn = np.zeros(n, np.float32)
    lg = np.log1p(np.maximum(mat.astype(np.float64), 0.0))
    for i in range(n):
        rs, ta_x, rn_x = [], [], []
        row = mat[i]
        for d in range(C.WINDOW, cut_col):
            if row[d] <= 0:
                continue
            w = lg[i, d - C.WINDOW:d]
            pos = w[row[d - C.WINDOW:d] > 0]
            if len(pos) < 8:
                continue
            rs.append(lg[i, d] - pos.mean())
            ta_x.append(anom_by_day[d])
            rn_x.append(rain_by_day[d])
        if len(rs) < 30:
            continue
        rs, ta_x, rn_x = np.array(rs), np.array(ta_x), np.array(rn_x)
        if rs.std() > 0 and ta_x.std() > 0:
            beta_ta[i] = np.corrcoef(rs, ta_x)[0, 1]
        wet, dry = rs[rn_x > 0], rs[rn_x == 0]
        if len(wet) >= 8 and len(dry) >= 8:
            beta_rn[i] = wet.mean() - dry.mean()
    return beta_ta, beta_rn


def beta_columns(meta, beta_ta, beta_rn, anom_by_day, rain_by_day):
    day_idx = meta[:, 0] + meta[:, 1]
    it = meta[:, 2]
    return np.column_stack([beta_ta[it] * anom_by_day[day_idx],
                            beta_rn[it] * rain_by_day[day_idx]]).astype(np.float32)


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    anom, rain = load_weather_daily()
    anom_by_day, rain_by_day = day_arrays(dates, anom, rain)

    keep_all = F.active_columns(include=("tb_visit",))
    names_all = F.active_names(include=("tb_visit",))
    base_sub = [i for i, n in enumerate(names_all) if n not in DROP3]
    assert len(base_sub) == 57

    out = {}
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        beta_ta, beta_rn = item_betas(mat, dates, cut_col, anom_by_day, rain_by_day)
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt0 = np.ascontiguousarray(Xtr[m][:, keep_all][:, base_sub])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv0 = np.ascontiguousarray(Xva[:, keep_all][:, base_sub])
        extra_tr = beta_columns(mtr[m], beta_ta, beta_rn, anom_by_day, rain_by_day)
        extra_va = beta_columns(mva, beta_ta, beta_rn, anom_by_day, rain_by_day)
        del Xtr, Xva
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)

        preds = {}
        for lab, use_beta in (("base", False), ("B", True)):
            nm = [names_all[i] for i in base_sub]
            Xt, Xv = Xt0, Xv0
            if use_beta:
                Xt = np.ascontiguousarray(np.hstack([Xt0, extra_tr]))
                Xv = np.ascontiguousarray(np.hstack([Xv0, extra_va]))
                nm = nm + ["wxb_ta", "wxb_rain"]
            cts = [c for c in F.CATEGORICAL if c in nm]
            ps = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=yt, feature_name=nm,
                                 categorical_feature=cts, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            preds[lab] = np.mean(ps, 0)

        cell_base = W * loss(yva, post(preds["base"], True, True))
        cell_b = W * loss(yva, post(preds["B"], True, True))
        total = float(cell_base.sum() - cell_b.sum())
        print(f"\n[{fname}] base prod {cell_base.sum():.5f} · B 이득 {total:+.5f} "
              f"(TB7 실측과 일치해야 함)", flush=True)

        rows = []
        for o in np.unique(mva[:, 0]):
            sel = mva[:, 0] == o
            g = float(cell_base[sel].sum() - cell_b[sel].sum())
            od = dates[int(o)]
            rows.append((str(od.date()), g))
            bar = "+" * int(round(max(g, 0) * 2000)) or ("-" * int(round(-g * 2000)))
            print(f"  origin {od.date()}  {g:+.5f}  {bar}", flush=True)
        gains = np.array([g for _, g in rows])
        pos_share = float((gains > 0).mean())
        top1 = float(gains.max() / total) if total > 0 else float("nan")
        print(f"  → 양수 origin 비율 {pos_share:.0%} · top-1 기여율 "
              f"{top1:.0%} (v27은 90%였다)", flush=True)
        out[fname] = {"total": total, "origins": rows,
                      "pos_share": pos_share, "top1_share": top1}
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print(f"\n저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
