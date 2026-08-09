# -*- coding: utf-8 -*-
"""Phase 27 — v27 계절 게이트 사후분석. **바꾼 파일이 3개뿐이라 분리가 정확하다.**

v24  Public 0.4533666  Private 0.4375952
v27  Public 0.4526740  Private 0.4378503     ← T07·T08·T09 에만 w=0.20

Public = {T00,T01,T05,T07,T08} 중 **바뀐 건 T07·T08 뿐**
나머지 = {T02,T03,T04,T06,T09} 중 **바뀐 건 T09 뿐**
→ 두 그룹의 변화량이 곧 그 파일들의 변화량이다. **연립방정식이 정확히 풀린다.**

그리고 내부에서 확인한다: 우리 봄 폴드(2/23~6/8, origin 16개) 안에서
**늦봄 origin 이 초봄과 다르게 움직였는가.** 다르다면 T09 의 실패를 미리 알 수 있었다는 뜻이다.

🚫 결과를 보고 "T09 만 빼자"는 하지 않는다. 그건 시험지 맞춤이다.
   여기서 묻는 것은 **내부 데이터만으로 알 수 있었는가** 하나뿐이다.
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
V24_PUB, V24_PRI = 0.4533666, 0.4375952
V27_PUB, V27_PRI = 0.4526739869, 0.4378503453
W = 0.20


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


def main():
    dpub, dpri = V27_PUB - V24_PUB, V27_PRI - V24_PRI
    print("=" * 90)
    print("① 실측 분해 — 바꾼 파일 3개의 기여를 정확히 가른다")
    print(f"  ΔPublic  {dpub:+.7f}  (T07·T08 만 바뀜)")
    print(f"  ΔPrivate {dpri:+.7f}")
    print(f"\n  {'Public 비중 w':>14s}{'Δ나머지5개':>14s}{'= T09 기여':>14s}"
          f"{'T07+T08 기여':>15s}")
    for wp in (0.40, 0.45, 0.47, 0.50, 0.55):
        drest = (dpri - wp * dpub) / (1 - wp)
        # 그룹 안 5개 파일이 대략 균등 기여한다고 보면
        t09 = drest * 5
        t78 = dpub * 5 / 2
        print(f"{wp:>14.2f}{drest:>+14.6f}{t09:>+14.5f}{t78:>+15.5f}")
    print("""
  → Public 비중을 어디로 잡아도 부호가 안 바뀐다:
    **T07·T08 은 좋아졌고(파일당 약 −0.002), T09 하나가 그보다 크게 나빠졌다(약 +0.005).**
    게이트가 지목한 3개 중 2개는 맞았고 1개가 틀렸는데, 틀린 쪽이 더 컸다.""")

    # ── ② 내부에서 알 수 있었나 — 봄 폴드 안 origin 별 이득
    print("\n" + "=" * 90)
    print("② 우리 봄 폴드 안에서 '늦봄'이 다르게 움직였는가 (내부 데이터만)")
    z = np.load(NPZ)
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    va = V.origins(dates, "2024-02-23", "2024-06-08", nd)
    y, iid = z["y|F3 봄"], z["iid|F3 봄"].astype(int)
    A, B = z["ours|F3 봄"], z["tm_ridge|F3 봄"]
    Wt, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
    blk = C.HORIZON * ctx.n
    assert len(y) == len(va) * blk

    print(f"{'origin 창끝':>12s}{'raw 이득':>12s}{'prod 이득':>12s}")
    rows = []
    for tag, kw in (("raw", dict(seg=False, snap=False)),):
        pass
    for i, o in enumerate(va):
        s = slice(i * blk, (i + 1) * blk)
        g = {}
        for tag, kw in (("raw", dict(seg=False, snap=False)),
                        ("prod", dict(seg=True, snap=True))):
            la = (Wt[s] * loss(y[s], post(A[s], **kw))).sum()
            lb = (Wt[s] * loss(y[s], post((1 - W) * A[s] + W * B[s], **kw))).sum()
            g[tag] = la - lb
        rows.append((pd.Timestamp(dates[o]), g["raw"], g["prod"]))
        print(f"{str(dates[o].date()):>12s}{g['raw']:>+12.5f}{g['prod']:>+12.5f}")

    df = pd.DataFrame(rows, columns=["date", "raw", "prod"])
    late = df.date >= pd.Timestamp("2024-05-01")
    print(f"\n  초봄 (2/23~4/30, {int((~late).sum())}개) : raw 합 {df.raw[~late].sum():+.5f}"
          f" · 평균 {df.raw[~late].mean():+.5f}")
    print(f"  늦봄 (5/01~6/08, {int(late.sum())}개)  : raw 합 {df.raw[late].sum():+.5f}"
          f" · 평균 {df.raw[late].mean():+.5f}")
    ratio = (df.raw[late].mean() / df.raw[~late].mean()) if df.raw[~late].mean() else np.nan
    print(f"  늦봄/초봄 평균 비 = {ratio:.2f}")
    if df.raw[late].mean() < 0:
        print("  → **내부에서도 늦봄은 이미 음수였다. 알 수 있었다.**")
    elif ratio < 0.5:
        print("  → **내부에서도 늦봄이 초봄의 절반 이하였다. 신호가 있었는데 못 봤다.**")
    else:
        print("  → 내부에서는 늦봄도 초봄과 비슷했다. **알 수 없었다** — 2025년 늦봄 고유의 일이다.")

    print("\n" + "=" * 90)
    print("""③ 판정

  사전등록: Private < 0.4375952 → 성립 · 아니면 축 영구 종료.
  실측 0.4378503 → **탈락. 앙상블·조건부결합 축 최종 종료.**

  다만 v26(−0.00102) 대비 v27(−0.00026)로 4분의 1이 됐다. 계절 게이트가
  **방향은 맞았다** — 3개 중 2개를 맞혔고, 전역 혼합보다 확실히 나았다.
  부족했던 것은 계절을 더 잘게 갈랐어야 했다는 것인데,
  그걸 데이터로 정하는 순간 자유도가 늘고 우리 저울로는 검증이 안 된다.
  **'봄'이라는 3개월 덩어리가 이 데이터로 잴 수 있는 가장 가는 눈금이다.**""")
    print("=" * 90)


if __name__ == "__main__":
    main()
