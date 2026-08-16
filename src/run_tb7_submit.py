# -*- coding: utf-8 -*-
"""Tier B — TB7 제출본 2종: x3(전면 적용) · x3s(봄 라우팅). run_tb3_submit.py와 동일 절차.

📌 사전등록 (experiments/log_tierb_TB7_차민.md 제출 절, 2026-08-16)
    x3  = x1(57열) + 품목 베타 2열(wxb_ta·wxb_rain) 전면 적용, 59열 전체학습.
          베타는 train 전체(겨울 1.5회 포함)로 추정 — FAR-겨울의 "여름 베타를 겨울에 적용"
          문제가 실전 구조에서는 완화된다는 가설의 직접 시험.
    x3s = 봄 라우팅: 예측 창의 중앙 목표일(4번째 날)이 [2/23, 6/8](= F3 폴드 정의,
          사후 조정 없음) 안이면 x3, 밖이면 기존 x1 raw(phase_tb3_visit_x1_raw.npy) 사용.
          근거: origin 분해(phase_tb7_spring_decomp.json)에서 봄 이득이 양수 81/75%·
          top-1 기여 21/17%로 고루 퍼짐 확인(v27의 90% 집중과 정반대).
    ⚠️ Tier B — 예측 대상일 실측 날씨 사용(상한선, x1의 실측 방문자수와 동일 전제).
    후처리는 v17 확정 구성 그대로: --seg 0.55,0.90,1.02 --t 1.8,10 --snap geom
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F

NT = os.cpu_count()
DROP = ["w_posmedian", "w_last14", "w_std"]
SEEDS = (42, 7, 2024, 913, 31)
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
OUT_X3 = os.path.join(C.EXPERIMENTS, "phase_tb7_x3_raw.npy")
OUT_X3S = os.path.join(C.EXPERIMENTS, "phase_tb7_x3s_raw.npy")
X1_RAW = os.path.join(C.EXPERIMENTS, "phase_tb3_visit_x1_raw.npy")
SPRING = ((2, 23), (6, 8))          # F3 폴드 정의 그대로 (사전등록, 사후 조정 금지)


def load_weather_daily():
    df = pd.read_csv(os.path.join(C.DATA, "tierb", "weather_icheon.csv"),
                     encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ta = df["ta_avg"].astype(float)
    anom = (ta - ta.rolling(15, center=True, min_periods=1).mean()).to_numpy()
    rain = (df["rn_day"].astype(float).to_numpy() >= 1.0).astype(float)
    return (dict(zip(df["date"], anom)), dict(zip(df["date"], rain)))


def item_betas_full(mat, dates, anom, rain):
    """train 전체로 품목 베타 추정 (TB7과 동일 정의, cut = 끝까지)."""
    n, nd = mat.shape
    a_day = np.zeros(nd, np.float32)
    r_day = np.zeros(nd, np.float32)
    for j, dt in enumerate(dates):
        if dt in anom:
            a_day[j] = anom[dt]
            r_day[j] = rain[dt]
    beta_ta = np.zeros(n, np.float32)
    beta_rn = np.zeros(n, np.float32)
    lg = np.log1p(np.maximum(mat.astype(np.float64), 0.0))
    for i in range(n):
        rs, ta_x, rn_x = [], [], []
        row = mat[i]
        for d in range(C.WINDOW, nd):
            if row[d] <= 0:
                continue
            w = lg[i, d - C.WINDOW:d]
            pos = w[row[d - C.WINDOW:d] > 0]
            if len(pos) < 8:
                continue
            rs.append(lg[i, d] - pos.mean())
            ta_x.append(a_day[d])
            rn_x.append(r_day[d])
        if len(rs) < 30:
            continue
        rs, ta_x, rn_x = np.array(rs), np.array(ta_x), np.array(rn_x)
        if rs.std() > 0 and ta_x.std() > 0:
            beta_ta[i] = np.corrcoef(rs, ta_x)[0, 1]
        wet, dry = rs[rn_x > 0], rs[rn_x == 0]
        if len(wet) >= 8 and len(dry) >= 8:
            beta_rn[i] = wet.mean() - dry.mean()
    return beta_ta, beta_rn, a_day, r_day


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    anom, rain = load_weather_daily()
    beta_ta, beta_rn, a_day, r_day = item_betas_full(mat, dates, anom, rain)
    print(f"베타 추정(전체 train, 겨울 포함): beta_ta 비영 {(beta_ta != 0).sum()}개 · "
          f"beta_rn 비영 {(beta_rn != 0).sum()}개", flush=True)

    base_keep = F.active_columns(include=("tb_visit",))
    base_names = F.active_names(include=("tb_visit",))
    sub = [i for i, n in enumerate(base_names) if n not in DROP]
    keep = [base_keep[i] for i in sub]
    names = [base_names[i] for i in sub] + ["wxb_ta", "wxb_rain"]
    cats = [c for c in F.CATEGORICAL if c in names]
    print(f"피처 {len(names)}개 (= x1 57 + 베타 2)", flush=True)
    assert len(names) == 59

    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    Xa, ya, ma = F.build_samples(mat, dates,
                                 list(range(C.WINDOW - 1, nd - C.HORIZON)), ctx)
    m = ya != 0
    day_idx = ma[m][:, 0] + ma[m][:, 1]
    it = ma[m][:, 2]
    extra_tr = np.column_stack([beta_ta[it] * a_day[day_idx],
                                beta_rn[it] * r_day[day_idx]]).astype(np.float32)
    Xt = np.ascontiguousarray(np.hstack([Xa[m][:, keep], extra_tr]))
    yt = np.log1p(np.maximum(ya[m], 1.0))
    del Xa

    tests, mids = [], []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        X, _, meta = F.build_samples(tmat, tdates, [C.WINDOW - 1], ctx,
                                     with_target=False)
        # 테스트 창은 날짜축이 28일뿐 — td는 인덱스가 아니라 타임스탬프로 계산
        origin = tdates[C.WINDOW - 1]
        h = meta[:, 1]
        itm = meta[:, 2]
        av = np.array([anom.get(origin + pd.Timedelta(days=int(k)), 0.0) for k in h],
                      dtype=np.float32)
        rv = np.array([rain.get(origin + pd.Timedelta(days=int(k)), 0.0) for k in h],
                      dtype=np.float32)
        ex = np.column_stack([beta_ta[itm] * av, beta_rn[itm] * rv]).astype(np.float32)
        assert np.isfinite(ex).all()
        tests.append(np.hstack([X[:, keep], ex]))
        mid = origin + pd.Timedelta(days=4)          # 예측 7일의 중앙 목표일
        mids.append(mid)
        cover = np.mean([origin + pd.Timedelta(days=int(k)) in anom
                         for k in range(1, 8)])
        print(f"  TEST_{t:02d} 목표 {str((origin+pd.Timedelta(days=1)).date())}"
              f"~{str((origin+pd.Timedelta(days=7)).date())} · 날씨 커버 {cover:.0%}",
              flush=True)
    Xte = np.ascontiguousarray(np.concatenate(tests, 0))
    print(f"학습 {Xt.shape[0]:,}행 · 테스트 {Xte.shape[0]:,}칸", flush=True)

    P = []
    for sd in SEEDS:
        ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                         categorical_feature=cats, free_raw_data=False)
        P.append(np.expm1(lgb.train(dict(PARAMS, seed=sd), ds,
                                    num_boost_round=ROUNDS).predict(Xte)))
        print(f"  시드 {sd} 완료 ({time.time()-t0:.0f}s)", flush=True)
        del ds
    raw_x3 = np.mean(P, 0)
    np.save(OUT_X3, raw_x3)

    # ---- x3s: 봄 라우팅 (창 중앙 목표일이 SPRING 안이면 x3, 밖이면 x1) ----
    raw_x1 = np.load(X1_RAW)
    assert raw_x1.shape == raw_x3.shape, (raw_x1.shape, raw_x3.shape)
    per = C.HORIZON * ctx.n                          # 테스트 파일당 행 수 (7×193)
    raw_x3s = raw_x1.copy()
    routed = []
    for t in range(C.N_TEST):
        md = (mids[t].month, mids[t].day)
        if SPRING[0] <= md <= SPRING[1]:
            raw_x3s[t * per:(t + 1) * per] = raw_x3[t * per:(t + 1) * per]
            routed.append(t)
    np.save(OUT_X3S, raw_x3s)
    print(f"\nx3s 라우팅: 봄 창 = TEST_{routed} (중앙 목표일 기준 [2/23, 6/8])")
    print(f"저장: {os.path.basename(OUT_X3)} · {os.path.basename(OUT_X3S)}")
    print("다음:")
    print(f"  python make_submission.py x3_wxbeta  --raw {OUT_X3} "
          "--seg 0.55,0.90,1.02 --t 1.8,10 --snap geom")
    print(f"  python make_submission.py x3s_spring --raw {OUT_X3S} "
          "--seg 0.55,0.90,1.02 --t 1.8,10 --snap geom")
    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
