# -*- coding: utf-8 -*-
"""Phase 10-b — 정수 스냅이 확정된 뒤, 남은 자유도를 다시 잰다.

배경 (Phase 10-a 실측)
    v9 0.451614 → **v12 0.448296 (+0.00332)**. 내부 OOF 예측 +0.00302 의 **1.10배**.
    → 후처리 축에서는 내부 저울이 거의 1:1 로 옮겨온다. ("×2" 는 보정 축 전용이었다.)

⭐ 스냅이 c1 을 먹어치웠다
    기하 스냅의 1↔2 경계는 √2 = 1.4142 다. `raw < t1` 구간은 `c1·raw` 인데,
    c1 < 0.7071 이면 raw < 2 인 모든 칸에서 `c1·raw < 1.4142` → **전부 1.0**.
    즉 **c1 은 완전히 무력화됐다** (0.55 든 0.40 이든 결과가 같다).
    → 남은 진짜 레버는 **t1 (어디까지를 그냥 1로 찍을 것인가)** 이다.

여기서 재는 것
    ① c1 무력화 확인   ② t1 스윕   ③ snapmax   ④ c2·c3 (스냅 격자 때문에 계단이 된다)
"""
import os
import sys

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

EXP = C.EXPERIMENTS
OOF = os.path.join(EXP, "phase8a_oof.npz")
FOLD = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]


def seg(raw, c=(0.55, 0.90, 1.02), t=(2.0, 10.0), floor=1.0):
    c1, c2, c3 = c
    t1, t2 = t
    p = np.where(raw < t1, c1 * raw, np.where(raw < t2, c2 * raw, c3 * raw))
    return np.maximum(p, floor)


def snap(p, mode="geom", maxv=np.inf, floor=1.0):
    if mode == "none":
        return p
    q = p.astype(np.float64).copy()
    m = q <= maxv
    if mode == "arith":
        q[m] = np.round(q[m])
    else:
        k = np.maximum(np.floor(q[m]), 1.0)
        q[m] = np.where(q[m] >= np.sqrt(k * (k + 1.0)), k + 1.0, k)
    return np.maximum(q, floor)


def main():
    ctx = F.Context()
    d = np.load(OOF)
    folds = []
    for i in range(4):
        p = d[f"ps{i}"]
        folds.append(dict(name=FOLD[i], p=(p.mean(0) if p.ndim == 2 else p),
                          y=d[f"y{i}"], iid=d[f"i{i}"]))

    def sc(f, p):
        return competition_score(f["y"], p, f["iid"], ctx.store_of_item,
                                 make_weights(1.0), ctx.n)

    def ev(**kw):
        cfg = dict(c=(0.55, 0.90, 1.02), t=(2.0, 10.0), mode="geom", maxv=np.inf)
        cfg.update(kw)
        v = [sc(f, snap(seg(f["p"], cfg["c"], cfg["t"]), cfg["mode"], cfg["maxv"]))
             for f in folds]
        return np.mean(v), v

    base, bv = ev()
    print("=" * 92)
    print("기준 = v12 구성 (seg 0.55/0.90/1.02 · 경계 2,10 · 기하 스냅 전구간)")
    print("  4폴드: " + " · ".join(f"{n} {x:.5f}" for n, x in zip(FOLD, bv))
          + f"   평균 {base:.5f}")
    print("  ※ 실측 대조: v12 합산 0.448296 (v9 0.451614 대비 +0.00332, 내부의 1.10배)")

    print("\n" + "=" * 92)
    print("① c1 이 정말 무력해졌는가 (기하 스냅 켠 상태)")
    print("=" * 92)
    for c1 in (0.30, 0.40, 0.55, 0.65, 0.70, 0.75, 0.85):
        m, _ = ev(c=(c1, 0.90, 1.02))
        print(f"  c1={c1:.2f}   평균 {m:.6f}   기준대비 {base-m:+.6f}"
              + ("   ← √2/2=0.7071 초과부터 갈린다" if abs(c1 - 0.75) < 1e-9 else ""))

    print("\n" + "=" * 92)
    print("② t1 스윕 — '어디까지 그냥 1로 찍을 것인가'. 스냅 뒤 남은 진짜 레버")
    print("=" * 92)
    print(f"  {'t1':>5s} " + "".join(f"{n:>12s}" for n in FOLD)
          + f"{'평균':>11s}{'기준대비':>11s}{'일관':>7s}")
    best = []
    for t1 in np.round(np.arange(1.6, 3.81, 0.2), 2):
        m, v = ev(t=(t1, 10.0))
        ok = sum(b - x > 0 for b, x in zip(bv, v))
        print(f"  {t1:5.1f} " + "".join(f"{x:12.5f}" for x in v)
              + f"{m:11.5f}{base-m:+11.5f}{ok:>5d}/4")
        best.append((base - m, t1, ok))

    print("\n" + "=" * 92)
    print("③ snapmax — 큰 값도 정수로 옮길 것인가")
    print("=" * 92)
    for mx in (3, 5, 10, 20, 50, np.inf):
        m, v = ev(maxv=mx)
        ok = sum(b - x > 0 for b, x in zip(bv, v))
        lab = "전구간" if not np.isfinite(mx) else f"p<={mx:g}"
        print(f"  {lab:>8s}  평균 {m:.5f}   기준대비 {base-m:+.6f}   일관 {ok}/4")

    print("\n" + "=" * 92)
    print("④ c2 · c3 — 스냅 격자 때문에 계단이 된다")
    print("=" * 92)
    print("  c2:", end="")
    for c2 in (0.80, 0.85, 0.90, 0.95, 1.00):
        m, _ = ev(c=(0.55, c2, 1.02))
        print(f"   {c2:.2f}→{base-m:+.5f}", end="")
    print("\n  c3:", end="")
    for c3 in (0.95, 1.00, 1.02, 1.05, 1.10, 1.18):
        m, _ = ev(c=(0.55, 0.90, c3))
        print(f"   {c3:.2f}→{base-m:+.5f}", end="")
    print()

    top = sorted(best, reverse=True)[:3]
    print("\n" + "=" * 92)
    print("t1 상위 3 (고립된 봉우리인지 이웃 값과 함께 볼 것):")
    for g, t1, ok in top:
        print(f"  t1={t1:.1f}  기준대비 {g:+.5f}  일관 {ok}/4")


if __name__ == "__main__":
    main()
