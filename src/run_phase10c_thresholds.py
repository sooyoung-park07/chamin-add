# -*- coding: utf-8 -*-
"""Phase 10-c — 후처리를 '임계 벡터'로 직접 파라미터화하고 **천장**을 잰다.

배경
    Phase 10-b 에서 기하 스냅이 c1 을 무력화했고(c1<0.7071 이면 전부 같은 함수),
    t1 을 2.0→1.8 로 옮겨 실측 +0.0021 을 얻었다(내부 +0.0022, 배율 0.81~0.97).
    → 지금 후처리는 실질적으로 **raw 를 정수로 보내는 계단 함수** 하나다.

⭐ 핵심 관찰 — 계단의 모양은 '임계 벡터' 가 전부다
    보정(c1,c2,c3) + 경계(t1,t2) + 하한 + 스냅 이 합쳐서 만드는 것은 결국
        raw ∈ [θ_k, θ_{k+1})  →  정수 k
    라는 **단조 계단**이다. 현재 값에서 유도되는 임계는 θ2=1.800 · θ3=2.722 · θ4=3.849 …
    배율을 우회해 **θ 를 직접 최적화**하면 이 축의 천장을 볼 수 있다.

⭐ 그리고 채점식은 칸별 가중합으로 **정확히** 분해된다
        Score = Σ_c W_c · loss(a_c, p_c),   W_c = w_s / (|I_s| · T_i) / Σw
    W_c 가 **A 에만 의존하고 P 와 무관**하기 때문이다(metrics.py: valid = a!=0 → T_i,
    NaN 품목 제외 → |I_s|, 가중 재정규화 → Σw). 이걸 쓰면 채점이 벡터 한 줄이 되어
    임계 최적화를 수만 번 돌릴 수 있다. **아래에서 competition_score 와 일치를 먼저 검증한다.**

규칙 준수
    · 자유도를 크게 늘리므로 **반드시 교차검증**한다(규칙 ④):
      가까움(F2·F3)에서 맞추고 멂(FAR 2개)에서 채점 / 그 반대. 낙관치와 정직치를 같이 본다.
    · 최적이 **넓은 고원**인지 본다(규칙 ⑦). 고립되면 노이즈다.
"""
import os

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

EXP = C.EXPERIMENTS
FOLD = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
NEAR, FARF = [0, 1], [2, 3]
KMAX = 14                      # θ2..θ_KMAX 를 최적화. 그 위는 기존 규칙 유지


def cell_weights(y, iid, store_of_item, n_items):
    """metrics.competition_score 와 동일한 가중치를 칸별로 편다."""
    valid = y != 0
    T = np.bincount(iid[valid], minlength=n_items).astype(np.float64)
    stores = sorted(set(np.asarray(store_of_item).tolist()))
    st = np.asarray(store_of_item)
    w = np.zeros(n_items)
    used_w = []
    for s in stores:
        m = (st == s) & (T > 0)
        if m.sum() == 0:
            continue
        used_w.append(1.0)
        w[m] = 1.0 / m.sum()                    # 1/|I_s|
    tot = float(len(used_w))                     # 균등 가중 → Σw = 업장 수
    W = np.zeros(len(y))
    W[valid] = w[iid[valid]] / T[iid[valid]] / tot
    return W, valid


def seg_snap(raw, c=(0.55, 0.90, 1.02), t=(1.8, 10.0)):
    c1, c2, c3 = c
    t1, t2 = t
    p = np.where(raw < t1, c1 * raw, np.where(raw < t2, c2 * raw, c3 * raw))
    p = np.maximum(p, 1.0)
    k = np.maximum(np.floor(p), 1.0)
    return np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k)


def induced_thetas(kmax=KMAX):
    """현행 규칙이 만드는 임계 θ_k (raw 축)."""
    g = np.arange(0.0, 60.0, 0.0002)
    v = seg_snap(g)
    th = []
    for k in range(2, kmax + 1):
        i = np.argmax(v >= k)
        th.append(g[i] if v[i] >= k else np.nan)
    return np.array(th)


def apply_thetas(raw, th, kmax=KMAX):
    """raw >= θ_k → 최소 k. θ_kmax 이상은 기존 규칙(값이 이미 kmax 이상)."""
    out = seg_snap(raw)                                  # 상단 구간은 그대로
    lo = raw < th[-1]
    v = np.ones(lo.sum())
    r = raw[lo]
    for k, t in enumerate(th, start=2):
        v[r >= t] = k
    out[lo] = v
    return out


def main():
    ctx = F.Context()
    d = np.load(os.path.join(EXP, "phase8a_oof.npz"))
    fo = []
    for i in range(4):
        y, iid = d[f"y{i}"], d[f"i{i}"]
        W, valid = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        fo.append(dict(name=FOLD[i], raw=d[f"ps{i}"].mean(0), y=y, iid=iid,
                       W=W, valid=valid))

    def fast(f, p):
        a = np.abs(f["y"]); q = np.abs(p)
        den = a + q
        t = np.zeros(len(a))
        m = f["valid"] & (den > 0)
        t[m] = 2.0 * np.abs(f["y"][m] - p[m]) / den[m]
        return float((f["W"] * t).sum())

    print("=" * 92)
    print("[0] 빠른 채점이 공식 채점과 일치하는가")
    print("=" * 92)
    for f in fo:
        p = seg_snap(f["raw"])
        a = competition_score(f["y"], p, f["iid"], ctx.store_of_item,
                              make_weights(1.0), ctx.n)
        b = fast(f, p)
        print(f"  {f['name']:<9s} 공식 {a:.9f}  빠름 {b:.9f}  차이 {abs(a-b):.2e}"
              + ("  일치" if abs(a - b) < 1e-9 else "  ⚠️ 불일치"))

    th0 = induced_thetas()
    base = np.array([fast(f, apply_thetas(f["raw"], th0)) for f in fo])
    print(f"\n  현행 임계 θ2..θ{KMAX}: " + " ".join(f"{x:.3f}" for x in th0))
    print(f"  현행 4폴드 평균 {base.mean():.6f}   (실측 대조: v17 합산 0.4459573)")

    # ── 좌표하강으로 θ 최적화 ────────────────────────────────────────
    def optimize(idx, th_init, rounds=4):
        th = th_init.copy()
        for _ in range(rounds):
            for j in range(len(th)):
                lo = th[j - 1] + 0.01 if j > 0 else 1.02
                hi = th[j + 1] - 0.01 if j < len(th) - 1 else th[j] * 1.6
                grid = np.linspace(lo, hi, 41)
                best, bv = th[j], np.inf
                for g in grid:
                    th[j] = g
                    v = np.mean([fast(fo[i], apply_thetas(fo[i]["raw"], th))
                                 for i in idx])
                    if v < bv:
                        bv, best = v, g
                th[j] = best
        return th

    print("\n" + "=" * 92)
    print("[1] 천장 (4폴드 전부에서 맞춤 — 낙관치)")
    print("=" * 92)
    th_all = optimize([0, 1, 2, 3], th0)
    s_all = np.array([fast(f, apply_thetas(f["raw"], th_all)) for f in fo])
    print("  최적 θ: " + " ".join(f"{x:.3f}" for x in th_all))
    print("  현행 θ: " + " ".join(f"{x:.3f}" for x in th0))
    print(f"  평균 {s_all.mean():.6f}   개선 {base.mean()-s_all.mean():+.6f}  ← 이 축의 천장")

    print("\n" + "=" * 92)
    print("[2] 교차검증 (2폴드에서 맞추고 나머지 2폴드에서 채점 — 정직치)")
    print("=" * 92)
    gains = []
    for fit, sco, tag in ((NEAR, FARF, "가까움→멂"), (FARF, NEAR, "멂→가까움")):
        th = optimize(fit, th0)
        b = np.mean([fast(fo[i], apply_thetas(fo[i]["raw"], th0)) for i in sco])
        a = np.mean([fast(fo[i], apply_thetas(fo[i]["raw"], th)) for i in sco])
        gains.append(b - a)
        print(f"  {tag}: 개선 {b-a:+.6f}")
        print("    맞춰진 θ: " + " ".join(f"{x:.3f}" for x in th))
    print(f"\n  교차검증 평균 {np.mean(gains):+.6f}   "
          f"과적합분 {(base.mean()-s_all.mean())-np.mean(gains):+.6f}")

    print("\n" + "=" * 92)
    print("[3] θ2 · θ3 만 단독으로 (자유도 2개, 고원 확인)")
    print("=" * 92)
    for name, j in (("θ2", 0), ("θ3", 1)):
        print(f"  {name} (현행 {th0[j]:.3f})")
        cur = th0[j]
        for g in np.round(np.linspace(cur * 0.75, cur * 1.30, 12), 3):
            th = th0.copy(); th[j] = g
            v = np.array([fast(f, apply_thetas(f["raw"], th)) for f in fo])
            ok = int((v < base).sum())
            print(f"    {g:6.3f}  평균 {v.mean():.6f}  개선 {base.mean()-v.mean():+.6f}"
                  f"  일관 {ok}/4")


if __name__ == "__main__":
    main()
