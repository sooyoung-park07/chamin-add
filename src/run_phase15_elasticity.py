# -*- coding: utf-8 -*-
"""Phase 15 — 창 탄력성 측정 (드리프트 취약성의 내부 정량화).

질문: **창이 c배 약해지면 예측도 c배 약해지는가?**
    연도 드리프트 가설(테스트 연도가 학습연도보다 약함 — 창 YoY 측정으로 확인)을 내부에서 정량화한다.

방법 (반사실 개입 — 관찰 회귀가 아님):
    1. 폴드 학습구간으로 모델을 학습 (원본 그대로)
    2. 검증 origin 의 피처를 두 번 만든다: 원본 mat / mat×c (창 유래 피처가 전부 일관되게 c배)
       달력·품목ID·가격은 그대로 → 모델의 '암기'는 옛 수준을 가리키고 '창'은 약해진 수준을 가리킴
    3. 탄력성 e = Δlog1p(예측) / log(c).  e=1 완전 창-추종 · e=0 완전 암기

해석: e 가 낮은 곳 = 드리프트에서 과대예측이 나는 곳. T05(스키 성수기) 과대와 대조한다.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

SEEDS = (42, 7, 2024)
NT = os.cpu_count()
PARAMS = dict(objective="regression_l1", metric="l1", learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=40, feature_fraction=0.65,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              verbosity=-1, num_threads=NT)
ROUNDS = 1000
CS = (0.85, 0.70)

FOLDS = [("F2 겨울", "2023-11-24", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-02-23", "2024-06-08")]


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    keep, names = F.active_columns(), F.active_names()
    cats = [c for c in F.CATEGORICAL if c in names]
    dvals = pd.DatetimeIndex(dates)

    for fname, cut, v0, v1 in FOLDS:
        cut_col = int(np.searchsorted(np.array(dates), pd.Timestamp(cut)))
        ctx.set_proxy(F.pick_proxy_items(mat, dates, cut_col, ctx.store_codes))
        trn = [o for o in range(C.WINDOW - 1, nd - C.HORIZON)
               if dates[o] < pd.Timestamp(cut)]
        va = V.origins(dates, v0, v1, nd)
        Xtr, ytr, _ = F.build_samples(mat, dates, trn, ctx)
        m = ytr != 0
        Xt = np.ascontiguousarray(Xtr[m][:, keep])
        yt = np.log1p(np.maximum(ytr[m], 1.0))
        del Xtr
        models = []
        for sd in SEEDS:
            ds = lgb.Dataset(Xt, label=yt, feature_name=names,
                             categorical_feature=cats, free_raw_data=False)
            models.append(lgb.train(dict(PARAMS, seed=sd), ds,
                                    num_boost_round=ROUNDS))
            del ds
        del Xt

        # 검증 피처: 원본과 스케일본
        preds = {}
        metas = None
        for c in (1.0,) + CS:
            mm = mat if c == 1.0 else mat * c
            Xva, yva, mva = F.build_samples(mm, dates, va, ctx)
            Xv = np.ascontiguousarray(Xva[:, keep])
            preds[c] = np.mean([np.expm1(md.predict(Xv)) for md in models], 0)
            if c == 1.0:
                metas = (yva, mva)
            del Xva, Xv
        yva, mva = metas
        W, valid = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        td = dvals[mva[:, 0] + mva[:, 1]]
        p0 = np.maximum(preds[1.0], 0.0)

        print("\n" + "=" * 92)
        print(f"[{fname}]  탄력성 e = Δlog1p(예측)/log(c)   (1=창추종 · 0=암기)")
        print("=" * 92)
        for c in CS:
            e_cell = (np.log1p(np.maximum(preds[c], 0.0)) - np.log1p(p0)) / np.log(c)
            ok = valid & (p0 > 0.3)
            e_all = float(np.average(e_cell[ok], weights=W[ok]))
            print(f"\n  c={c}  전체 탄력성 {e_all:.3f}")
            st = ctx.store_of_item[mva[:, 2]]
            rows = []
            for s in sorted(set(st.tolist())):
                mm2 = ok & (st == s)
                if mm2.sum() < 50:
                    continue
                rows.append((s, float(np.average(e_cell[mm2], weights=W[mm2]))))
            rows.sort(key=lambda x: x[1])
            print("   업장별(낮을수록 암기 의존): "
                  + " · ".join(f"{s} {v:.2f}" for s, v in rows))
            mo = td.month.values
            rows = []
            for month in sorted(set(mo.tolist())):
                mm2 = ok & (mo == month)
                if mm2.sum() < 200:
                    continue
                rows.append((month, float(np.average(e_cell[mm2], weights=W[mm2]))))
            print("   월별: " + " · ".join(f"{month}월 {v:.2f}" for month, v in rows))
            q = np.quantile(p0[ok], [0.25, 0.5, 0.75])
            lab = [f"~{q[0]:.1f}", f"~{q[1]:.1f}", f"~{q[2]:.1f}", "이상"]
            bins = np.digitize(p0, q)
            rows = []
            for b in range(4):
                mm2 = ok & (bins == b)
                rows.append((lab[b], float(np.average(e_cell[mm2], weights=W[mm2]))))
            print("   예측크기별: " + " · ".join(f"{k} {v:.2f}" for k, v in rows))


if __name__ == "__main__":
    main()
