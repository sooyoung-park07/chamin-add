# -*- coding: utf-8 -*-
"""데이터 무결성 검증 — 새 환경/새 데이터 사본에서 가장 먼저 1회 실행.

config.EXPECTED의 기준값과 대조해서 하나라도 어긋나면 FAIL.
(open 폴더 삭제 사고 이후, '데이터가 진짜 그 데이터인가'를 코드로 확인하는 장치)
"""
import numpy as np
import pandas as pd

import config as C
import dataio as D


def main() -> int:
    fails = []

    def check(name, cond, detail=""):
        print(("  [OK]  " if cond else "  [FAIL]") + f" {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            fails.append(name)

    print("=== train.csv ===")
    tr = D.load_train()
    check("행 수", len(tr) == C.EXPECTED["train_rows"], f"{len(tr):,}")
    check("품목 수", tr["key"].nunique() == C.EXPECTED["n_items"], str(tr["key"].nunique()))
    check("영업장 수", tr["store"].nunique() == C.EXPECTED["n_stores"], str(tr["store"].nunique()))
    check("기간 시작", str(tr["date"].min().date()) == C.EXPECTED["train_start"], str(tr["date"].min().date()))
    check("기간 끝", str(tr["date"].max().date()) == C.EXPECTED["train_end"], str(tr["date"].max().date()))
    check("결측 없음", tr["qty"].notna().all())
    check("전 품목×전 일자 조합 완전", len(tr) == tr["key"].nunique() * tr["date"].nunique(),
          f"{tr['key'].nunique()}×{tr['date'].nunique()}")
    check("군집 매핑 완전", set(tr["store"].unique()) == set(C.STORE_CLUSTER.keys()))

    print("=== test ===")
    items = D.item_order()
    prev_end = None
    for i in range(C.N_TEST):
        te = D.load_test(i)
        days = te["date"].nunique()
        ok_days = days == C.WINDOW
        ok_items = set(te["key"]) == set(items)
        if prev_end is not None:
            # 이전 테스트의 예측구간(+1~+7일) 직후에 다음 창이 시작하는지
            gap = (te["date"].min() - prev_end).days
            check(f"TEST_{i:02d} 연속성", gap == C.HORIZON + 1, f"직전 창 끝과 {gap}일 간격")
        check(f"TEST_{i:02d} 28일/193품목", ok_days and ok_items, f"{days}일")
        prev_end = te["date"].max()

    print("=== sample_submission ===")
    sub = D.load_submission_template()
    check("shape", sub.shape == C.EXPECTED["submission_shape"], str(sub.shape))
    check("행 라벨 70개", list(sub.iloc[:, 0]) ==
          [f"TEST_{t:02d}+{h}일" for t in range(C.N_TEST) for h in range(1, C.HORIZON + 1)])

    print("=== labels ===")
    labels = D.load_labels()
    check("라벨 193개", len(labels) == C.EXPECTED["n_items"], str(len(labels)))
    prices = [D.label_of(labels, k, "price_krw", 0) for k in items]
    check("가격 전부 존재(>0)", all(p and p > 0 for p in prices))

    print("=== 알려진 통계 재확인 ===")
    check("0 비율 ≈ 52.6%", abs((tr["qty"] == 0).mean() - 0.526) < 0.01,
          f"{(tr['qty'] == 0).mean():.3f}")
    check("음수 존재(환불)", (tr["qty"] < 0).any(), f"min={tr['qty'].min():.0f}")
    hwadam = tr[tr["store"].isin(["화담숲주막", "화담숲카페"])]
    winter = hwadam[hwadam["date"].dt.month.isin([12, 1, 2])]
    check("화담숲 겨울 완전 0", (winter["qty"] == 0).all(), f"{len(winter)}행")

    print()
    if fails:
        print(f"결론: FAIL {len(fails)}건 — {fails}")
        return 1
    print("결론: 무결성 검증 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
