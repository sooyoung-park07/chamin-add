# -*- coding: utf-8 -*-
"""Phase 22-b — 22 의 라우팅 신호를 **TEST 에서 독립 검증**한다.

22 는 내부 창 58개에서 4/4 같은 방향인 서술자 넷을 찾았다:
    near_boundary(−0.455) · zero(+0.413) · cv(+0.408) · level(−0.355)
넷 다 "조용하고 희소하고 변동 크고 시즌 경계에 가까운 창일수록 혼합이 이득"을 가리킨다.

**이게 진짜인지 가릴 수 있는 독립 증거가 하나 있다.**
실측에서 혼합은 Public 5개 파일 {T00,T01,T05,T07,T08} 에서 이득(−0.0016),
나머지 5개 {T02,T03,T04,T06,T09} 에서 손해(+0.0033) 였다.

그러니 이렇게 시험한다:
  · 규칙은 **내부 58개 창에서만** 나왔다 (TEST 점수를 안 봤다).
  · 그 규칙으로 TEST 10개 창의 순위를 매긴다.
  · 상위 5개가 {T00,T01,T05,T07,T08} 와 얼마나 겹치는가?

⚠️ 이건 '검증'이지 '적합'이 아니다. 결과를 보고 규칙을 손보면 그 순간 시험지 맞춤이 된다.
   그래서 합성 점수 형태(z 4개 등가중)를 **여기 미리 고정**하고 한 번만 돌린다.
   자유 파라미터 0개 — 문턱도 안 맞춘다. 5:5 분할이 정해져 있으니 상위 5개를 그냥 자른다.

무작위 기대: 10개에서 5개를 뽑아 정답 5개와 겹치는 수의 분포 (초기하)
    겹침 0개 0.4% · 1개 9.9% · 2개 39.7% · 3개 39.7% · 4개 9.9% · 5개 0.4%
→ **4개 이상 겹치면 상위 10.3%, 5개면 상위 0.4%.** 3개는 우연이다.
"""
import os
from math import comb

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
from run_phase22_window_signal import descriptors

PUBLIC = {"TEST_00", "TEST_01", "TEST_05", "TEST_07", "TEST_08"}
KEYS = ["cv", "zero", "level", "near_boundary"]
SIGN = {"cv": +1, "zero": +1, "level": -1, "near_boundary": -1}   # 22 에서 나온 부호


def main():
    ref = pd.read_csv(os.path.join(C.EXPERIMENTS, "phase22_window_signal.csv"),
                      encoding="utf-8-sig")
    mu = {k: ref[k].mean() for k in KEYS}
    sd = {k: ref[k].std() for k in KEYS}
    print("기준 분포 (내부 창 58개) — TEST 를 표준화하는 데만 쓴다")
    for k in KEYS:
        print(f"  {k:<15s} 평균 {mu[k]:>8.4f} · 표준편차 {sd[k]:>8.4f} · 부호 {SIGN[k]:+d}")

    ctx = F.Context()
    items = ctx.items
    n_store = int(np.max(ctx.store_codes)) + 1

    rec = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, items)
        assert tmat.shape[1] == 28
        # descriptors 는 (mat, dates, o) 를 받으므로 창의 마지막 인덱스를 넘긴다
        d = descriptors(tmat, tdates, C.WINDOW - 1, ctx.store_codes, n_store)
        z = sum(SIGN[k] * (d[k] - mu[k]) / sd[k] for k in KEYS)
        rec.append(dict(test=f"TEST_{t:02d}", score=z,
                        **{k: d[k] for k in KEYS},
                        start=str(pd.Timestamp(tdates[-1]).date())))
    df = pd.DataFrame(rec).sort_values("score", ascending=False).reset_index(drop=True)
    df["예측"] = ["혼합"] * 5 + ["v24 단독"] * 5
    df["실제"] = np.where(df.test.isin(PUBLIC), "이득군(Public)", "손해군")

    print("\n" + "=" * 100)
    print("TEST 10개 창 — 내부 규칙이 매긴 순위 (위쪽일수록 '혼합이 이득일 것')")
    print(f"{'순위':>4s} {'파일':<9s}{'창끝':>12s}{'합성z':>9s}"
          f"{'cv':>8s}{'zero':>8s}{'level':>8s}{'경계거리':>9s}  {'예측':<9s}{'실제'}")
    for i, r in df.iterrows():
        hit = "✓" if (r["예측"] == "혼합") == (r.test in PUBLIC) else "✗"
        print(f"{i+1:>4d} {r.test:<9s}{r.start:>12s}{r.score:>9.2f}"
              f"{r.cv:>8.3f}{r.zero:>8.3f}{r.level:>8.2f}{r.near_boundary:>9.0f}"
              f"  {r['예측']:<9s}{r['실제']} {hit}")

    top5 = set(df.head(5).test)
    k = len(top5 & PUBLIC)
    p_ge = sum(comb(5, j) * comb(5, 5 - j) for j in range(k, 6)) / comb(10, 5)
    print("\n" + "=" * 100)
    print(f"상위 5개 = {sorted(top5)}")
    print(f"실제 이득군 = {sorted(PUBLIC)}")
    print(f"→ **겹침 {k}/5**  ·  우연히 이만큼 이상 겹칠 확률 {100*p_ge:.1f}%")
    if k >= 4:
        print("   ✅ 신호가 TEST 로 옮겨왔다. 라우팅 규칙이 실재할 가능성이 있다.")
        print("      다만 이건 1비트짜리 증거다 — 채택하려면 사전등록 후 제출 1회가 필요하다.")
    elif k == 3:
        print("   ⚠️ 우연 범위(39.7%)다. 내부 상관은 있었으나 TEST 로 옮겨왔다는 증거가 없다.")
    else:
        print("   ❌ 오히려 반대쪽이다. 내부 58개 창의 상관은 TEST 에 대해 무의미하다.")
        print("      → 라우팅 신호 없음. **조건부 결합 축도 닫힌다.**")
    print("=" * 100)

    out = os.path.join(C.EXPERIMENTS, "phase22b_holdout.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {os.path.basename(out)}")


if __name__ == "__main__":
    main()
