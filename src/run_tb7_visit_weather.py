# -*- coding: utf-8 -*-
"""Tier B — TB7: 방문자수 조건부 날씨. run_tb3_visitors.py와 동일한 저울, base만 x1(=v24+방문자수 57열)로 올린다.

📌 사전등록 (2026-08-16, 결과 보기 전)
    배경: TB2 계열(날씨 5회)은 전부 v24(54열, 방문자수 없음) 위에서 검증됐고, TB3(방문자수)와
    같이 넣어본 적은 한 번도 없다. 날씨→방문자수→매출 인과 사슬에서 방문자수가 이미 피처로
    있다면 날씨의 "방문 수요" 몫은 방문자수가 흡수하고, 남는 몫은
      (a) 광주시 전체 방문자수 ↔ 리조트 발길의 괴리를 날씨가 메우는 부분
      (b) 메뉴 단위 반응(더우면 아이스↑ 국밥↓ — 방문자수가 같아도 뭘 사먹는지가 달라짐)
    뿐이다. 트리는 방문자수로 분기한 가지 안에서 날씨로 재분기할 수 있으므로(편상관의 트리 아날로그),
    이 실험은 "방문자수 조건부 날씨 몫의 존재"를 직접 잰다.

    팔 A (TB7a, 60열): base(57) + 날씨 원값 3열(ta_avg·rn_day·sd_max, 예측 대상일 td 실측,
        게이트 없음 — TB2 방식 전 업장). 조건부 몫이 (a)+(b) 어느 쪽이든 있으면 잡힌다.
    팔 B (TB7b, 59열): base(57) + 품목별 민감도 베타 2열 — (b)를 1차 피처로 직접 노출:
        wxb_ta   = beta_ta[i] × ta_anom(td)    (품목별 온도민감도 × 당일 기온 이상치)
        wxb_rain = beta_rn[i] × is_rain(td)    (품목별 강수효과   × 당일 비 여부)
      beta는 폴드마다 학습 컷 이전 데이터로만 계산한다(누출 방지, pick_proxy_items와 동일 관례):
        resid_i(d) = log1p(s_i(d)) − 직전 28일 중 양수판매일의 log1p 평균 (양수판매일만, 최소 8일)
        beta_ta[i] = corr(resid_i, ta_anom)      — 관측 30일 미만이면 0
        beta_rn[i] = mean(resid|비) − mean(resid|맑음) — 비/맑음 각 8일 미만이면 0
        ta_anom(d) = ta_avg(d) − 중심 15일 이동평균(기온의 계절 성분 제거 — 겨울 스키·여름 피서
                     같은 '계절 그 자체'는 base 캘린더가 이미 가졌으므로, "평년 대비 유난히
                     덥거나 춥다"만 남긴다). is_rain(d) = rn_day ≥ 1.0mm.

    ⚠️ 두 팔 다 td 당일 실측 날씨를 쓰는 상한선 실험(TB2·TB3와 동일 전제 — 배포하려면 예보 필요).
    A·B는 각각 base와만 짝비교하며 절대 합치지 않는다(한 번에 후보 하나 규약의 병렬 적용).

    예상치: A 평균 +0.000 ~ +0.005 — 방문자수가 날씨의 수요 몫을 다 흡수했다면 0 근처
        (TB2 봄 이득 +0.016과 TB3 봄 이득 +0.016이 거의 같은 크기였다는 게 그 방증).
        B도 같은 범위. B가 A보다 크면 메뉴 반응(b)이 실재한다는 증거.
        FAR류 폴드는 열 개수 부작용(규칙 ⑯, TB2e에서 확인)으로 마이너스 가능성을 사전에
        기록해둔다. 이번 실험의 1차 관심은 게이트 통과가 아니라 "조건부 몫의 존재 여부"이므로
        게이트 판정과 폴드별 해석을 병기한다.

    게이트(기록용, 기존과 동일): prod 평균 > +0.0015 AND prod 4/4 일관 AND raw 4/4 양수.

    🔎 실행 전 자가검증(TB4b feature_names 사고 교훈): base가 정확히 57열인지, A가 정확히
    +3열인지, B의 추가 2열이 전부 유한하고 0이 아닌 값을 실제로 갖는지 출력·확인한다.
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

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2023-11-24", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22")]
OUT = os.path.join(C.EXPERIMENTS, "phase_tb7_visit_weather.json")

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    ROUNDS, SEEDS, FOLDS = 60, (42,), FOLDS[:1]
    print("⚠️ SMOKE 모드: 1폴드·1시드·60라운드 — 배선 검증 전용, 수치는 무의미", flush=True)

BASE_LABEL = "L0 base(57=54+visit)"
ARM_A = "TB7a +weather3(60)"
ARM_B = "TB7b +itembeta2(59)"


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


def load_weather_daily():
    """이천 ASOS 실측 → (기온 이상치, 비 여부) 딕셔너리 {Timestamp: float}."""
    df = pd.read_csv(os.path.join(C.DATA, "tierb", "weather_icheon.csv"),
                     encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ta = df["ta_avg"].astype(float)
    anom = (ta - ta.rolling(15, center=True, min_periods=1).mean()).to_numpy()
    rain = (df["rn_day"].astype(float).to_numpy() >= 1.0).astype(float)
    return (dict(zip(df["date"], anom)), dict(zip(df["date"], rain)))


def day_arrays(dates, anom, rain):
    """행렬의 날짜축 인덱스 → 그날의 (기온 이상치, 비 여부) 배열. 날씨 없는 날은 0."""
    a = np.zeros(len(dates), np.float32)
    r = np.zeros(len(dates), np.float32)
    for j, dt in enumerate(dates):
        if dt in anom:
            a[j] = anom[dt]
            r[j] = rain[dt]
    return a, r


def item_betas(mat, dates, cut_col, anom_by_day, rain_by_day):
    """품목별 온도민감도·강수효과 — 학습 컷 이전 데이터로만 (사전등록 정의 그대로)."""
    n, _ = mat.shape
    beta_ta = np.zeros(n, np.float32)
    beta_rn = np.zeros(n, np.float32)
    lg = np.log1p(mat.astype(np.float64))
    n_obs = np.zeros(n, np.int32)
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
        n_obs[i] = len(rs)
        if len(rs) < 30:
            continue
        rs, ta_x, rn_x = np.array(rs), np.array(ta_x), np.array(rn_x)
        if rs.std() > 0 and ta_x.std() > 0:
            beta_ta[i] = np.corrcoef(rs, ta_x)[0, 1]
        wet, dry = rs[rn_x > 0], rs[rn_x == 0]
        if len(wet) >= 8 and len(dry) >= 8:
            beta_rn[i] = wet.mean() - dry.mean()
    return beta_ta, beta_rn, n_obs


def beta_columns(meta, beta_ta, beta_rn, anom_by_day, rain_by_day):
    """샘플 행(meta=[origin,horizon,item]) → 추가 2열. td = dates[o+h] (날짜축 연속 — build_samples와 동일 가정)."""
    day_idx = meta[:, 0] + meta[:, 1]
    it = meta[:, 2]
    col_ta = beta_ta[it] * anom_by_day[day_idx]
    col_rn = beta_rn[it] * rain_by_day[day_idx]
    return np.column_stack([col_ta, col_rn]).astype(np.float32)


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    anom, rain = load_weather_daily()
    anom_by_day, rain_by_day = day_arrays(dates, anom, rain)
    print(f"날씨 매칭: 기온이상치 비영 {np.count_nonzero(anom_by_day)}/{nd}일 · "
          f"비온날 {int(rain_by_day.sum())}일", flush=True)

    keep_all = F.active_columns(include=("tb_visit", "tb_weather"))
    names_all = F.active_names(include=("tb_visit", "tb_weather"))
    wx_idx = [i for i, n in enumerate(names_all) if n in F.TB_WEATHER_KEYS]
    base_sub = [i for i, n in enumerate(names_all)
                if n not in DROP3 and n not in F.TB_WEATHER_KEYS]
    a_sub = sorted(set(base_sub + wx_idx))
    print(f"열 검증 — base {len(base_sub)}열(=57? {len(base_sub) == 57}) · "
          f"A {len(a_sub)}열(=60? {len(a_sub) == 60})", flush=True)
    assert len(base_sub) == 57 and len(a_sub) == 60

    res = {}
    betas_log = {}
    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))

        # ---- 품목별 베타 (컷 이전 데이터만) + 상식 검증 출력 ----
        beta_ta, beta_rn, n_obs = item_betas(mat, dates, cut_col, anom_by_day, rain_by_day)
        est = n_obs >= 30
        betas_log[fname] = {
            "estimated_items": int(est.sum()),
            "top_ta": [(ctx.items[i], round(float(beta_ta[i]), 3))
                       for i in np.argsort(-beta_ta)[:5]],
            "bottom_ta": [(ctx.items[i], round(float(beta_ta[i]), 3))
                          for i in np.argsort(beta_ta)[:5]],
            "top_rain": [(ctx.items[i], round(float(beta_rn[i]), 3))
                         for i in np.argsort(-beta_rn)[:3]],
            "bottom_rain": [(ctx.items[i], round(float(beta_rn[i]), 3))
                            for i in np.argsort(beta_rn)[:3]]}
        print(f"\n[{fname}] 베타 추정 품목 {est.sum()}/{len(ctx.items)}", flush=True)
        print("  더울수록↑:", ", ".join(f"{k}({v:+.2f})" for k, v in betas_log[fname]["top_ta"]))
        print("  더울수록↓:", ", ".join(f"{k}({v:+.2f})" for k, v in betas_log[fname]["bottom_ta"]))
        print("  비오면↑ :", ", ".join(f"{k}({v:+.2f})" for k, v in betas_log[fname]["top_rain"]))
        print("  비오면↓ :", ", ".join(f"{k}({v:+.2f})" for k, v in betas_log[fname]["bottom_rain"]), flush=True)

        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, mtr = F.build_samples(mat, dates, trn, ctx)
        Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
        m = ytr != 0
        Xt0 = np.ascontiguousarray(Xtr[m][:, keep_all])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        Xv0 = np.ascontiguousarray(Xva[:, keep_all])
        extra_tr = beta_columns(mtr[m], beta_ta, beta_rn, anom_by_day, rain_by_day)
        extra_va = beta_columns(mva, beta_ta, beta_rn, anom_by_day, rain_by_day)
        del Xtr, Xva
        assert np.isfinite(extra_tr).all() and np.isfinite(extra_va).all()
        nz = (extra_tr != 0).mean(0)
        print(f"  베타 열 비영 비율 — train wxb_ta {nz[0]:.1%} · wxb_rain {nz[1]:.1%}", flush=True)
        assert nz.max() > 0.05, "추가 열이 사실상 전부 0 — 배선 오류 의심 (TB4b 교훈)"
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)

        arms = [(BASE_LABEL, base_sub, None),
                (ARM_A, a_sub, None),
                (ARM_B, base_sub, (extra_tr, extra_va))]
        for lab, sub, extra in arms:
            nm = [names_all[i] for i in sub]
            Xt = Xt0[:, sub]
            Xv = Xv0[:, sub]
            if extra is not None:
                Xt = np.hstack([Xt, extra[0]])
                Xv = np.hstack([Xv, extra[1]])
                nm = nm + ["wxb_ta", "wxb_rain"]
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
        del Xt0, Xv0, extra_tr, extra_va

        b = res[(fname, BASE_LABEL)]
        for lab in (ARM_A, ARM_B):
            c = res[(fname, lab)]
            print(f"  [{fname:<8s}] {lab:<22s} base prod {b['prod']:.5f}"
                  f" · 차이 prod {b['prod']-c['prod']:+.5f} · raw {b['raw']-c['raw']:+.5f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump({"betas": betas_log,
                   "res": {f"{k[0]}|{k[1]}": v for k, v in res.items()}},
                  open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("\n" + "=" * 96)
    for cand in (ARM_A, ARM_B):
        done = [f for f in FOLDS if (f[0], cand) in res]
        gp = [res[(f[0], BASE_LABEL)]["prod"] - res[(f[0], cand)]["prod"] for f in done]
        gr = [res[(f[0], BASE_LABEL)]["raw"] - res[(f[0], cand)]["raw"] for f in done]
        print(f"\n📌 {cand}  (base 대비, +가 좋음)")
        print("  prod " + " ".join(f"{f[0]}:{g:+.5f}" for f, g in zip(done, gp)))
        print("  raw  " + " ".join(f"{f[0]}:{g:+.5f}" for f, g in zip(done, gr)))
        ok = (np.mean(gp) > 0.0015) and (sum(g > 0 for g in gp) == len(done) == 4) \
            and (sum(g > 0 for g in gr) == 4)
        print(f"  prod 평균 {np.mean(gp):+.5f} · 일관 {sum(g>0 for g in gp)}/{len(done)}"
              f" · raw 평균 {np.mean(gr):+.5f} · 일관 {sum(g>0 for g in gr)}/{len(done)}"
              f"  → {'✅ 통과' if ok else '❌ 탈락'}")
    print("=" * 96)
    print(f"저장: {os.path.basename(OUT)} · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
