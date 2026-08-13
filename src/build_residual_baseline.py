# -*- coding: utf-8 -*-
"""베이스 모델(v24, 54피처)의 예측 잔차를 날짜별로 뽑아낸다.

목적: 새 후보 피처가 "이미 아는 정보의 재인코딩"이 아니라 "진짜 새 정보"인지 확인하려면,
원본 수요가 아니라 이 잔차와 상관을 봐야 한다. 베이스가 못 맞춘 부분에만 후보가 반응하면
진짜 추가 정보, 원본 수요와만 상관 있으면 이미 흡수된 정보일 가능성이 크다.

학습: cut 이전 전부. 예측: cut ~ 2024-06-08 (연속 구간, ~10개월치 잔차 확보).
cut을 가장 이를 걸로 잡는 이유 — FAR류 폴드에서 지금까지 신호가 제일 크게 나왔던 구간과
겹치게 해서, 데이터 넉넉한 잔차 샘플을 얻기 위함.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F

CUT = "2023-08-25"
END = "2024-06-08"
SEED = 42
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=os.cpu_count())
ROUNDS = 1000
OUT = os.path.join(C.EXPERIMENTS, "tierb_residual_daily.csv")


def main():
    t0 = time.time()
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep = F.active_columns()
    names = F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]

    cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(CUT)))
    ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
    trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
           if dates[o] < pd.Timestamp(CUT)]
    va = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
          if pd.Timestamp(CUT) <= dates[o] <= pd.Timestamp(END)]

    Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
    Xva, yva, mva = F.build_samples(mat, dates, va, ctx)
    m = ytr != 0
    Xt = np.ascontiguousarray(Xtr[m][:, keep])
    yt = np.log1p(np.maximum(ytr[m], 1.0))
    Xv = np.ascontiguousarray(Xva[:, keep])
    del Xtr, Xva

    ds = lgb.Dataset(Xt, label=yt, feature_name=names, categorical_feature=cats,
                      free_raw_data=False)
    pred = np.expm1(lgb.train(dict(PARAMS, seed=SEED), ds, num_boost_round=ROUNDS)
                     .predict(Xv))
    print(f"학습·예측 완료 ({time.time()-t0:.0f}s)")

    # td(=origin+h)별로 리조트 전체(전 영업장 합계) 실제/예측 집계
    origins, hs = mva[:, 0].astype(int), mva[:, 1].astype(int)
    tds = np.array([dates[o] + pd.Timedelta(days=h) for o, h in zip(origins, hs)])
    df = pd.DataFrame({"td": tds, "actual": yva, "pred": pred})
    daily = df.groupby("td").sum()
    daily["residual"] = daily["actual"] - daily["pred"]
    daily["resid_ratio"] = daily["actual"] / daily["pred"].clip(lower=1.0)
    daily.reset_index().to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} · {len(daily)}일치 잔차 · 총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()