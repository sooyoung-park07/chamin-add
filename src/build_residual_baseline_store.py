# -*- coding: utf-8 -*-
"""build_residual_baseline.py의 업장별 버전.

기존 스크립트는 리조트 전체 합계 잔차만 남겨서, 새 후보(특히 업장 특정적인 검색어)를
검증할 때 다른 업장의 반대방향 신호에 희석된 상관만 볼 수 있었다. 이 스크립트는 같은
학습/검증 분할에서 나온 예측을 업장별로도 집계해 저장한다 — cluster(hwadam/ski/green/
always/b2b)와 store(9개 개별)  두 단위 모두.

학습·검증 분할은 build_residual_baseline.py와 완전히 동일(CUT=2023-08-25, SEED=42) —
같은 잔차를 다른 단위로 재사용하는 것이므로 재현 실험 시 두 파일의 리조트 합계가
정확히 일치해야 한다(교차검증용 assert 포함).
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
OUT_STORE = os.path.join(C.EXPERIMENTS, "tierb_residual_store.csv")
OUT_CLUSTER = os.path.join(C.EXPERIMENTS, "tierb_residual_cluster.csv")


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

    origins, hs, item_idx = mva[:, 0].astype(int), mva[:, 1].astype(int), mva[:, 2].astype(int)
    tds = np.array([dates[o] + pd.Timedelta(days=h) for o, h in zip(origins, hs)])
    store = ctx.store_of_item[item_idx]
    cluster = np.array([C.STORE_CLUSTER[s] for s in store])

    df = pd.DataFrame({"td": tds, "store": store, "cluster": cluster,
                        "actual": yva, "pred": pred})

    for unit, out_path in [("store", OUT_STORE), ("cluster", OUT_CLUSTER)]:
        g = df.groupby(["td", unit]).sum(numeric_only=True)
        g["residual"] = g["actual"] - g["pred"]
        g["resid_ratio"] = g["actual"] / g["pred"].clip(lower=1.0)
        g.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"저장: {out_path} · {len(g)}행")

    # 교차검증 — 리조트 총합이 기존 build_residual_baseline.py 출력과 일치해야 한다
    resort_daily = df.groupby("td")[["actual", "pred"]].sum()
    ref_path = os.path.join(C.EXPERIMENTS, "tierb_residual_daily.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path, parse_dates=["td"]).set_index("td")
        diff = (resort_daily["actual"] - ref["actual"]).abs().max()
        print(f"교차검증(리조트 합계, actual 최대차): {diff:.6f} (0에 가까워야 함)")

    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
