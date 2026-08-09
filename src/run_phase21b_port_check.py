# -*- coding: utf-8 -*-
"""Phase 21-b — 이식판 검증. **이게 통과해야 21-c 가 의미를 가진다.**

`model_ridge_tm.py` 는 팀원 스크립트를 우리 하네스용으로 뜯어낸 것이다.
뜯는 과정에서 조용히 틀렸다면 뒤따르는 잔차 상관·앙상블 판정이 전부 쓰레기가 된다.
그래서 **같은 조건(전체학습 → 10개 테스트 창)** 에서 원본과 예측을 대조한다.

기대: 완전 일치는 안 된다. origin 목록이 하나 다르다(원본 index 28~, 우리 27~).
     학습행이 193×7 = 1,351 행(0.2%) 많다. 그 정도 차이만 나야 정상이다.
판정 (사전 고정):
     ✅ 로그공간 상관 r >= 0.999 AND 중앙 절대 로그차 <= 0.01
     ❌ 그 밖 → 이식 오류를 먼저 찾는다
"""
import os
import time

import numpy as np
import pandas as pd

import config as C
import dataio as D
import model_ridge_tm as TM

TM_LOG = os.path.join(C.ROOT, "teammate_repo", "outputs", "final_prediction_log.csv")
OUT = os.path.join(C.EXPERIMENTS, "phase21b_port_raw.npy")


def main():
    t0 = time.time()
    items = D.item_order()
    ctx = TM.TMContext(items)
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, items)
    nd = mat.shape[1]

    trn = list(range(C.WINDOW - 1, nd - C.HORIZON))
    print(f"학습 origin {len(trn)}개 (원본 497개, 우리 {len(trn)}개 — index 27 vs 28 차이)",
          flush=True)

    num, cat, y, item_idx = TM.build_frames(ctx, mat, dates, trn)
    print(f"학습 사례 {len(y):,} / 양수 {int((y > 0).sum()):,}  ({time.time()-t0:.0f}s)",
          flush=True)

    # 테스트 10개 창
    te_num, te_cat = [], []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, items)
        assert tmat.shape[1] == 28, f"TEST_{t:02d} 28일 아님"
        nu, ca = TM.build_from_window(ctx, tdates, tmat)
        te_num.append(nu)
        te_cat.append(ca)
    te_num = np.concatenate(te_num)
    te_cat = np.concatenate(te_cat)

    out = TM.fit_predict(ctx, num, cat, y, item_idx, te_num, te_cat)
    print(f"설계행렬 {out['design_shape']}  (원본 (319311, 3201))  "
          f"({time.time()-t0:.0f}s)", flush=True)
    np.save(OUT, out["blend"])

    # 원본과 대조
    df = pd.read_csv(TM_LOG, encoding="utf-8-sig")
    piv = df.pivot_table(index=["test", "horizon"], columns="item_id",
                         values="pred_ridge", aggfunc="first").reindex(columns=items)
    ref = np.concatenate([piv.loc[(f"TEST_{t:02d}", h)].to_numpy(float)
                          for t in range(C.N_TEST) for h in range(1, C.HORIZON + 1)])

    mine = out["ridge"]
    lo = lambda v: np.log1p(np.maximum(v, 0.0))
    r = float(np.corrcoef(lo(mine), lo(ref))[0, 1])
    dlog = np.abs(lo(mine) - lo(ref))
    med, p99 = float(np.median(dlog)), float(np.quantile(dlog, 0.99))

    print("\n" + "=" * 62)
    print("이식판 vs 원본 (ridge 단독, 같은 테스트 창)")
    print(f"  로그공간 상관   r = {r:.6f}          (기준 >= 0.999)")
    print(f"  절대 로그차     중앙 {med:.5f} · 99% {p99:.5f}   (기준 중앙 <= 0.01)")
    print(f"  중앙값          이식 {np.median(mine):.4f} / 원본 {np.median(ref):.4f}")
    ok = (r >= 0.999) and (med <= 0.01)
    print(f"\n  → {'✅ 이식 검증 통과. 21-c 진행 가능' if ok else '❌ 불일치. 이식 오류를 먼저 찾을 것'}")
    print("=" * 62)
    print(f"총 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
