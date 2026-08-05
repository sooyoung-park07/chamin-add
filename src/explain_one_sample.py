# -*- coding: utf-8 -*-
"""학습 샘플 한 줄이 실제로 어떻게 만들어지는지 눈으로 확인 (설명·교육용)."""
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F

ITEM = "담하_공깃밥"
HORIZON_PICK = 3


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    i = ctx.items.index(ITEM)

    # origin = 창의 마지막 날. 2024-05-01이 창 끝이 되도록 고름
    o = [k for k, d in enumerate(dates) if d == pd.Timestamp("2024-05-01")][0]
    lo = o - C.WINDOW + 1
    win_dates = dates[lo:o + 1]
    win_vals = mat[i, lo:o + 1]
    target_date = dates[o + HORIZON_PICK]
    target = mat[i, o + HORIZON_PICK]

    DOW = ["월", "화", "수", "목", "금", "토", "일"]
    print("=" * 78)
    print(f"품목: {ITEM}")
    print(f"창(28일): {win_dates[0].date()} ~ {win_dates[-1].date()}")
    print(f"예측 대상: +{HORIZON_PICK}일 = {target_date.date()}({DOW[target_date.dayofweek]})"
          f"  → 정답 {target:.0f}개")
    print("=" * 78)

    print("\n[주어진 28일 판매량]")
    for wk in range(4):
        seg = slice(wk * 7, wk * 7 + 7)
        ds = win_dates[seg]
        vs = win_vals[seg]
        print("  " + " ".join(f"{DOW[d.dayofweek]}{v:>4.0f}" for d, v in zip(ds, vs)))

    same_dow = [(d, v) for d, v in zip(win_dates, win_vals)
                if d.dayofweek == target_date.dayofweek]
    print(f"\n  예측일과 같은 요일({DOW[target_date.dayofweek]})만 뽑으면: "
          + ", ".join(f"{d.strftime('%m/%d')}={v:.0f}" for d, v in same_dow))

    # 실제 파이프라인으로 피처 생성
    X, y, meta = F.build_samples(mat, dates, [o], ctx)
    row = (meta[:, 1] == HORIZON_PICK) & (meta[:, 2] == i)
    x = X[row][0]
    fn = F.feature_names()

    print("\n[이 한 줄의 피처 43개]  ← 모델이 실제로 보는 것")
    for gname, keys in F.FEATURE_GROUPS.items():
        desc = {"win": "창 전체 통계", "dow": "같은 요일 통계",
                "ctx": "맥락(지난주 같은요일·영업장·horizon)",
                "cal": "예측 대상일의 달력·도메인",
                "item": "품목 고유 속성(시간 무관)"}[gname]
        print(f"\n  ── {gname}: {desc}")
        for k in keys:
            v = x[fn.index(k)]
            print(f"     {k:<16s} {v:>10.3f}")

    print(f"\n  → 타깃 y = {y[row][0]:.0f}   (학습 시엔 log1p({y[row][0]:.0f})="
          f"{np.log1p(y[row][0]):.3f} 를 맞추도록 학습)")

    print("\n" + "=" * 78)
    print("이런 줄이 총 몇 개 만들어지나")
    print("=" * 78)
    n_origin = len(range(C.WINDOW - 1, mat.shape[1] - C.HORIZON))
    print(f"  origin(창 끝날) 후보     : {n_origin}개  (2023-01-28 ~ 2024-06-08)")
    print(f"  × horizon                : {C.HORIZON}개  (+1일 ~ +7일)")
    print(f"  × 품목                   : {ctx.n}개")
    print(f"  = 전체 학습 줄            : {n_origin * C.HORIZON * ctx.n:,}개")
    Xa, ya, _ = F.build_samples(mat, dates,
                                list(range(C.WINDOW - 1, mat.shape[1] - C.HORIZON)), ctx)
    print(f"  이 중 정답이 0이 아닌 줄만 : {int((ya != 0).sum()):,}개  ← 실제 학습에 사용")
    print(f"  (정답 0인 줄 {int((ya == 0).sum()):,}개는 버림 — 채점에서 제외되는 행이라)")


if __name__ == "__main__":
    main()
