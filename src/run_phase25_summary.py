# -*- coding: utf-8 -*-
"""Phase 25 — 앙상블 축 전체를 숫자 하나로 모은다. 설명용.

묻는 것은 두 개뿐이다.
  ① 애초에 걸려 있던 상금이 얼마였나 (이론 천장)
  ② 우리가 본 숫자들은 각각 무엇이었나 (내부 +0.0018, 실전 −0.0010)

이론. 두 예측의 로그오차를 (1−w):w 로 섞으면
    σ(w)² = (1−w)²σ₁² + w²σ₂² + 2w(1−w)ρσ₁σ₂
최소값은 닫힌 형태로 나온다 (Bates–Granger 1969):
    w* = (1 − ρk)/(1 + k² − 2ρk),   k = σ₂/σ₁
    σ(w*)²/σ₁² = k²(1 − ρ²)/(1 + k² − 2ρk)
그리고 w* > 0 조건은 **ρ < 1/k**.
"""
import os

import numpy as np
from scipy.optimize import brentq

import config as C
import features as F
from run_phase10c_thresholds import cell_weights

NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")
FOLDS = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]


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


def wstat(w, x):
    m = np.sum(w * x) / np.sum(w)
    return m, np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))


def var_ratio(rho, k):
    return k * k * (1 - rho * rho) / (1 + k * k - 2 * rho * k)


def main():
    z = np.load(NPZ)
    ctx = F.Context()

    print("=" * 94)
    print("① 걸려 있던 상금 — 두 오차를 최적으로 섞었을 때 오차가 몇 % 줄어드나")
    print(f"{'폴드':<10s}{'σ우리':>8s}{'σ그들':>8s}{'k':>7s}{'ρ':>8s}"
          f"{'손익분기 1/k':>12s}{'w*':>8s}{'σ 감소':>9s}")
    rows = []
    for f in FOLDS:
        y, iid = z[f"y|{f}"], z[f"iid|{f}"].astype(int)
        A, B = z[f"ours|{f}"], z[f"tm_ridge|{f}"]
        W, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        sc = y != 0
        lg = lambda v: np.log1p(np.maximum(v, 0.0))
        a = lg(np.maximum(np.abs(y[sc]), 1.0))
        r1, r2 = lg(np.maximum(A[sc], 1.0)) - a, lg(B[sc]) - a
        ww = W[sc]
        _, s1 = wstat(ww, r1)
        _, s2 = wstat(ww, r2)
        m1, _ = wstat(ww, r1)
        m2, _ = wstat(ww, r2)
        rho = float(np.sum(ww * (r1 - m1) * (r2 - m2)) / np.sum(ww) / (s1 * s2))
        k = s2 / s1
        ws = (1 - rho * k) / (1 + k * k - 2 * rho * k)
        red = 100 * (1 - np.sqrt(max(var_ratio(rho, k), 0.0)))
        print(f"{f:<10s}{s1:>8.4f}{s2:>8.4f}{k:>7.3f}{rho:>8.4f}"
              f"{1/k:>12.4f}{ws:>+8.3f}{red:>8.2f}%")
        rows.append((rho, k, s1, s2))

    rho_m = float(np.mean([r[0] for r in rows]))
    k_m = float(np.mean([r[1] for r in rows]))
    red_m = 100 * (1 - np.sqrt(var_ratio(rho_m, k_m)))
    print(f"{'평균':<10s}{'':>16s}{k_m:>7.3f}{rho_m:>8.4f}{1/k_m:>12.4f}"
          f"{(1-rho_m*k_m)/(1+k_m*k_m-2*rho_m*k_m):>+8.3f}{red_m:>8.2f}%")

    print(f"\n  → 최적으로 섞어도 오차가 **{red_m:.2f}% 줄어드는 게 전부**다.")
    print(f"    우리 점수 ~0.49 에서 그건 대략 **{0.49*red_m/100:.4f}** 이고,")
    print(f"    실제로는 SMAPE 가 포화해서 그보다 작다 (실측 raw 천장 아래 ③ 참조).")
    print(f"    우리 분해능(2σ) = 0.003. **상금이 눈금보다 작다.**")

    # ── 얼마나 달랐어야 했나
    print("\n" + "=" * 94)
    print("② 이겼으려면 무엇이 얼마나 달랐어야 하나 (상금 0.003 = 눈금 1칸 기준)")
    target = 1 - (0.003 / 0.49)                       # 필요한 σ 감소 배율
    try:
        rho_need = brentq(lambda r: np.sqrt(var_ratio(r, k_m)) - target, 0.0, rho_m)
        print(f"  · ρ 를 낮춘다면 : 0.87 이 아니라 **ρ ≤ {rho_need:.3f}** 이어야 했다")
    except ValueError:
        print("  · ρ 만으로는 도달 불가")
    try:
        k_need = brentq(lambda kk: np.sqrt(var_ratio(rho_m, kk)) - target, 1.0001, k_m)
        gap = 100 * (k_need - 1)
        print(f"  · 팀원 모델이 더 셌다면 : k ≤ {k_need:.3f} "
              f"(**우리보다 {gap:.1f}% 이내**. 실제는 {100*(k_m-1):.1f}% 열세)")
    except ValueError:
        print(f"  · k 만으로는 도달 불가 (ρ={rho_m:.3f} 에서는 어떤 k 도 부족)")

    # ── 실측 곡선
    print("\n" + "=" * 94)
    print("③ 우리가 본 숫자 세 개의 정체")
    grid = np.round(np.arange(-0.10, 0.31, 0.025), 4)
    for pname, kw in (("raw (하한만)", dict(seg=False, snap=False)),
                      ("실제 (seg+스냅)", dict(seg=True, snap=True))):
        cur = []
        for w in grid:
            s = np.mean([float((cell_weights(z[f"y|{f}"], z[f"iid|{f}"].astype(int),
                                             ctx.store_of_item, ctx.n)[0]
                                * loss(z[f"y|{f}"], post(
                                    (1 - w) * z[f"ours|{f}"] + w * z[f"tm_ridge|{f}"], **kw))).sum())
                          for f in FOLDS])
            cur.append(s)
        cur = np.array(cur)
        i0 = int(np.argmin(np.abs(grid)))
        g = cur[i0] - cur
        best = int(np.argmax(g))
        print(f"  [{pname:<14s}] 최적 w={grid[best]:+.3f} · 천장 {g[best]:+.5f}"
              f"  ·  w=0.10 에서 {g[np.argmin(np.abs(grid-0.10))]:+.5f}")

    print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  이론 천장 (raw)        약 +0.0002   ← 진짜 상금                       │
  │  내부 측정 (seg+스냅)   +0.00184     ← 9배 부풀려짐. 정수 스냅의 이산 점프  │
  │  실전 (Private)         −0.00102     ← 같은 점프가 TEST 창에선 반대로     │
  └──────────────────────────────────────────────────────────────────────┘
  세 숫자 전부 '참값 +0.0002 주위에서 ±0.001~0.003 흔들리는 양자화 잡음' 하나로 설명된다.""")

    # ── 있었던 신호
    print("\n" + "=" * 94)
    print("④ 그래도 실재했던 신호 — 크기와 함께")
    z2 = {f: dict(y=z[f"y|{f}"], A=z[f"ours|{f}"], B=z[f"tm_ridge|{f}"],
                  iid=z[f"iid|{f}"].astype(int)) for f in FOLDS}
    f = "F3 봄"
    d = z2[f]
    W, _ = cell_weights(d["y"], d["iid"], ctx.store_of_item, ctx.n)
    lA = loss(d["y"], post(d["A"], seg=False, snap=False)) * W
    lB = loss(d["y"], post(d["B"], seg=False, snap=False)) * W
    wt = np.bincount(d["iid"], weights=W, minlength=ctx.n)
    sa = np.bincount(d["iid"], weights=lA, minlength=ctx.n)
    sb = np.bincount(d["iid"], weights=lB, minlength=ctx.n)
    ok = wt > 0
    nb = int((sb[ok] < sa[ok]).sum())
    print(f"  · 잔차 상관 ρ = {rho_m:.3f} — 트리끼리(XGB 0.965 · Cat 0.944)보다 낮다. **다양성은 진짜다.**")
    print(f"  · 품목 {int(ok.sum())}개 중 **{nb}개({100*nb/ok.sum():.0f}%)에서 팀원 모델이 낫다.**")
    print(f"    이 정보를 완벽히 알면 천장이 +0.019 — 우리 후보정 전체 이득(+0.0172)급이다.")
    print(f"  · 그런데 그 우위의 계절 간 지속성이 p=0.20 이라 **미리 알 방법이 없다.**")
    print(f"  · 우리 후처리는 남의 예측에도 이식된다 (팀원 모델 Private 0.4957 → 0.4885).")
    print("=" * 94)


if __name__ == "__main__":
    main()
