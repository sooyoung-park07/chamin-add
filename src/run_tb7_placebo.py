# -*- coding: utf-8 -*-
"""Tier B — TB7 부속: FAR-겨울 플라시보 검정. run_tb7_visit_weather.py와 동일한 저울.

📌 사전등록 (experiments/log_tierb_TB7_차민.md 부속 절, 2026-08-16)
    질문: TB7의 FAR-겨울 손해(A -0.02110 / B -0.02483)가 날씨 "내용" 탓인가,
    "열이 늘어난 것 자체"(규칙 ⑯) 탓인가. 정보가 0인 대조군 3종이 같은 손해를 내는지 잰다.
      N3 : 날 단위 iid 정규난수 3열 (A와 같은 구조·열수, 정보 0)
      R3 : 실제 날씨 3열을 날짜축 +180일 롤 (날씨의 분포·매끄러움 유지, 정렬만 파괴)
      N2 : B 구조 모사 — 품목계수×날난수 1열 + 품목계수×날베르누이(p=0.235) 1열
    예상: 열 부작용이 주범이면 N3·R3 ≈ -0.021 / N2 ≈ -0.025 부근.
         날씨 내용이 해로운 거면 플라시보는 -0.005~-0.009(역대 패턴)에 그침.
    난수 시드 20260816 고정(재현 가능). base(57)·폴드·짝시드·라운드는 TB7과 동일.
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
RNG_SEED = 20260816

FOLDS = [("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]
OUT = os.path.join(C.EXPERIMENTS, "phase_tb7_placebo.json")

BASE_LABEL = "L0 base(57)"


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


def weather_by_day(dates):
    """실측 날씨 3종을 행렬 날짜축에 정렬한 (nd, 3) 배열 — R3의 롤 원본."""
    df = pd.read_csv(os.path.join(C.DATA, "tierb", "weather_icheon.csv"),
                     encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    lut = {d: (a, b, c) for d, a, b, c in
           zip(df["date"], df["ta_avg"], df["rn_day"], df["sd_max"])}
    out = np.zeros((len(dates), 3), np.float32)
    for j, dt in enumerate(dates):
        if dt in lut:
            out[j] = lut[dt]
    return out


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    n_items = len(ctx.items)

    rng = np.random.default_rng(RNG_SEED)
    noise_day = rng.normal(size=(nd, 3)).astype(np.float32)            # N3
    rolled = np.roll(weather_by_day(dates), 180, axis=0)                # R3
    g_day = rng.normal(size=nd).astype(np.float32)                      # N2 열1의 날 성분
    b_day = (rng.random(nd) < 0.235).astype(np.float32)                 # N2 열2의 날 성분
    coef1 = rng.normal(0.0, 0.15, n_items).astype(np.float32)
    coef2 = rng.normal(0.0, 0.15, n_items).astype(np.float32)

    def extra_cols(meta, kind):
        day = meta[:, 0] + meta[:, 1]
        it = meta[:, 2]
        if kind == "N3":
            return noise_day[day]
        if kind == "R3":
            return rolled[day]
        if kind == "N2":
            return np.column_stack([coef1[it] * g_day[day],
                                    coef2[it] * b_day[day]]).astype(np.float32)
        raise ValueError(kind)

    keep_all = F.active_columns(include=("tb_visit",))
    names_all = F.active_names(include=("tb_visit",))
    base_sub = [i for i, n in enumerate(names_all) if n not in DROP3]
    print(f"열 검증 — base {len(base_sub)}열(=57? {len(base_sub) == 57})", flush=True)
    assert len(base_sub) == 57

    ARMS = [(BASE_LABEL, None, []),
            ("N3 날난수3(60)", "N3", ["nz_d1", "nz_d2", "nz_d3"]),
            ("R3 날씨롤180(60)", "R3", ["wx_ta_r", "wx_rn_r", "wx_sd_r"]),
            ("N2 구조난수2(59)", "N2", ["nzb_1", "nzb_2"])]

    res = {}
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt0 = np.ascontiguousarray(Xtr[m][:, keep_all])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv0 = np.ascontiguousarray(Xva[:, keep_all])
        mtr_m = mtr[m]
        del Xtr, Xva
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)

        for lab, kind, extra_names in ARMS:
            nm = [names_all[i] for i in base_sub]
            Xt = Xt0[:, base_sub]
            Xv = Xv0[:, base_sub]
            if kind is not None:
                et, ev = extra_cols(mtr_m, kind), extra_cols(mva, kind)
                assert np.isfinite(et).all() and np.isfinite(ev).all()
                nz = (et != 0).mean(0)
                print(f"  {lab} 추가열 비영 비율: {np.round(nz, 3)}", flush=True)
                assert nz.max() > 0.05
                Xt = np.hstack([Xt, et])
                Xv = np.hstack([Xv, ev])
                nm = nm + extra_names
            Xt = np.ascontiguousarray(Xt)
            Xv = np.ascontiguousarray(Xv)
            cts = [c for c in F.CATEGORICAL if c in nm]
            ps = []
            for sd in SEEDS:
                ds = lgb.Dataset(Xt, label=yt, feature_name=nm,
                                 categorical_feature=cts, free_raw_data=False)
                ps.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                             num_boost_round=ROUNDS).predict(Xv)))
                del ds
            p = np.mean(ps, 0)
            res[(fname, lab)] = {
                "n": len(nm),
                "prod": float((W * loss(yva, post(p, True, True))).sum()),
                "raw": float((W * loss(yva, post(p, False, False))).sum())}
            del Xt, Xv, ps
            if kind is not None:
                b = res[(fname, BASE_LABEL)]
                c = res[(fname, lab)]
                print(f"  [{fname}] {lab:<18s} base prod {b['prod']:.5f}"
                      f" · 차이 prod {b['prod']-c['prod']:+.5f}"
                      f" · raw {b['raw']-c['raw']:+.5f}  ({time.time()-t0:.0f}s)",
                      flush=True)
        json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
                  open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("📌 TB7 플라시보 판정 — TB7 실측(같은 폴드·같은 base·같은 짝시드):")
    print("   실제 날씨 A(+3열) prod -0.02110 · raw -0.02472")
    print("   실제 베타 B(+2열) prod -0.02483 · raw -0.02486")
    b = res[("FAR-겨울", BASE_LABEL)]
    for lab, kind, _ in ARMS[1:]:
        c = res[("FAR-겨울", lab)]
        print(f"   {lab:<18s} prod {b['prod']-c['prod']:+.5f} · raw {b['raw']-c['raw']:+.5f}")
    print("=" * 90)
    print(f"저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
