# -*- coding: utf-8 -*-
"""Phase 23-b — 라우팅을 **전역 가중치와 같은 잣대로** 비교한다. 그리고 스냅을 벗긴다.

23-a 에서 두 가지가 드러났다.
  ① **관측된 앙상블 이득의 대부분이 후처리 인공물이다.**
     w=0.10 이득: 하한만 −0.00033 / seg만 +0.00114 / seg+스냅 +0.00184
     즉 raw 수준에서는 이미 손해인데, 구간배율과 정수스냅을 거치면서 이득으로 보였다.
  ② 교차 라우팅이 봄에서 +0.00129 를 냈다 — 23-a 스크립트는 이걸 "이득"이라 판정했는데
     **비교 대상이 틀렸다.** 라우팅의 경쟁자는 'A 단독'이 아니라 **'전역 w'** 다.
     전역 w 를 이길 때만 라우팅에 값어치가 있다.

그래서 여기서 셋을 같은 규율로 세운다 (전부 겨울에서 고르고 봄에서 채점 — 정직판):
    · A 단독            (w=0, 파라미터 0개)
    · 전역 w            (겨울에서 고른 스칼라 1개)
    · 품목별 라우팅      (겨울에서 고른 w_i 193개)
  + 반칙 대조군: 봄에서 고른 오라클 (천장)
그리고 후처리 3종(하한만 / seg만 / seg+스냅) 각각에서 반복한다.

⭐ 자유도가 1 → 193 으로 늘 때 성적이 어떻게 되는지가 이 표의 전부다.
"""
import os

import numpy as np

import config as C
import features as F
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
POSTS = [("하한만 (raw)", dict(snap=False, seg=False)),
         ("seg만", dict(snap=False, seg=True)),
         ("seg+스냅 (실제 파이프라인)", dict(snap=True, seg=True))]


def post(raw, snap=True, seg=True):
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
    z = np.load(NPZ)
    ctx = F.Context()
    n = ctx.n
    FIT, SCORE = "F2 겨울", "F3 봄"          # 고르는 폴드 / 채점하는 폴드

    data = {}
    for f in (FIT, SCORE):
        y, iid = z[f"y|{f}"], z[f"iid|{f}"].astype(int)
        W, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        data[f] = dict(y=y, iid=iid, W=W, A=z[f"ours|{f}"], B=z[f"tm_ridge|{f}"])

    print(f"고르는 폴드 = {FIT} · 채점 폴드 = {SCORE}  (품목 축 n={n})")
    for pname, kw in POSTS:
        it = {}
        for f in (FIT, SCORE):
            d = data[f]
            it[f] = {w: np.bincount(d["iid"],
                                    weights=loss(d["y"], post(
                                        (1 - w) * d["A"] + w * d["B"], **kw)) * d["W"],
                                    minlength=n) for w in GRID}
        fit, sco = it[FIT], it[SCORE]
        base = sco[0.0].sum()

        # 전역 w — 겨울에서 고른다
        gw = min(GRID, key=lambda w: fit[w].sum())
        g_hon = sco[gw].sum()
        g_orc = min(sco[w].sum() for w in GRID)

        # 품목별 라우팅 — 겨울에서 고른다
        pick = np.array([min(GRID, key=lambda w: fit[w][i]) for i in range(n)])
        r_hon = sum(sco[pick[i]][i] for i in range(n))
        r_orc = sum(min(sco[w][i] for w in GRID) for i in range(n))

        print("\n" + "=" * 92)
        print(f"[{pname}]")
        print(f"{'방식':<28s}{'자유도':>7s}{'봄 점수':>11s}{'A단독 대비':>12s}{'전역w 대비':>12s}")
        rows = [("A 단독 (w=0)", 0, base),
                (f"전역 w (겨울에서 고름 → {gw})", 1, g_hon),
                ("품목별 라우팅 (겨울에서 고름)", n, r_hon),
                ("· 전역 w 오라클 (반칙)", 1, g_orc),
                ("· 품목별 오라클 (반칙)", n, r_orc)]
        for nm, dof, s in rows:
            print(f"{nm:<28s}{dof:>7d}{s:>11.5f}{base-s:>+12.5f}{g_hon-s:>+12.5f}")
        cap = (base - r_orc)
        got = (base - r_hon)
        print(f"  라우팅이 천장의 {100*got/cap if cap else 0:.0f}% 를 회수 "
              f"(천장 {cap:+.5f} → 실제 {got:+.5f})")
        print(f"  ⭐ 라우팅 − 전역w = {g_hon - r_hon:+.5f} "
              f"({'라우팅 승' if r_hon < g_hon else '**전역 w 승 — 자유도 193개가 손해**'})")
        print(f"  선택 분포: " + " ".join(f"{w}:{int((pick==w).sum())}"
                                       for w in GRID if (pick == w).any()))

    print("\n" + "=" * 92)
    print("""결론 읽는 법
  · '하한만' 열이 **모델 수준의 진실**이다. seg·스냅은 그 위에 얹힌 변환일 뿐이고,
    그 변환이 이득의 크기를 바꾼다면 그건 모델 이야기가 아니라 **양자화 이야기**다.
  · 라우팅이 전역 w 를 못 이기면, 자유도 193개를 쓴 대가가 순손실이라는 뜻이다.
    이 경우 '구분해서 합치기'는 원리적으로 닫힌다 — 신호가 없어서가 아니라
    **신호보다 추정오차가 커서**다.""")
    print("=" * 92)


if __name__ == "__main__":
    main()
