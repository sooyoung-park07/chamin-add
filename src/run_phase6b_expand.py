# -*- coding: utf-8 -*-
"""Phase 6-b — 로그공간 '펴기' 보정. Phase 6-a ④번 진단의 후속.

━━ 6-a에서 나온 것 ━━
  ① 하한: 4개 폴드 **만장일치로 1.00이 최적**. 사용자 가설(1~2 사이)은 기각.
  ② 전역 배율: 평균 최적 0.96(+0.0012)이나 **폴드별로 1.01/0.98/0.98/0.80 로 흩어짐** → 신뢰 못 함.
  ④ **구간별 최적 배율이 체계적으로 기울어져 있다:**

        예측 1.19 → 최적배율 0.60 (실측중앙 1.0, 예측중앙 1.19 = 과대예측)
        예측 7.16 → 최적배율 1.00
        예측 30.1 → 최적배율 1.17 (실측중앙 36.0, 예측중앙 30.1 = 과소예측)
        예측 73.1 → 최적배율 1.10 (실측중앙 85.0)

  → **모델 예측이 안쪽으로 압축돼 있다.** 작은 건 과대, 큰 건 과소.
    전역 배율(전부 같은 방향)로는 못 잡는 게 당연하다. 방향이 구간마다 반대니까.

━━ 그래서 이 실험 ━━
압축을 푸는 가장 단순한 형태 — **로그공간에서 중심 기준으로 늘린다**:

    log1p(p') = m + k · ( log1p(p) − m )        k > 1 이면 펴짐
    p' = max( expm1(...), 1.0 )

**자유도가 k 하나뿐**이라 과적합 위험이 거의 없다(구간별 배율 10개를 따로 맞추는 것과 대비).
로그공간을 쓰는 이유: SMAPE 는 log(y/p) 에 대해 대칭이므로 그 공간이 자연스럽다.

비교군으로 3구간 버전(자유도 3)도 같이 재서, 자유도를 늘릴 값어치가 있는지 본다.
**판정은 4개 폴드 일관성** — 한둘에서만 좋으면 버린다. 재학습 없음(저장된 OOF 사용, 수 초).
"""
import os
import json

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

OOF = os.path.join(C.EXPERIMENTS, "phase6a_oof.npz")
NAMES = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
KIND = ["가까움", "가까움", "멂", "멂"]
FLOOR = 1.0
SIGMA2 = 0.0032


def main():
    ctx = F.Context()
    z = np.load(OOF)
    folds = [dict(name=NAMES[i], kind=KIND[i], p=z[f"p{i}"].astype(np.float64),
                  y=z[f"y{i}"], iids=z[f"i{i}"]) for i in range(4)]

    def sc(d, p):
        return competition_score(d["y"], np.maximum(p, FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    base = [sc(d, d["p"]) for d in folds]
    b_avg = float(np.mean(base))
    print("=" * 96)
    print("Phase 6-b — 로그공간 '펴기' 보정")
    print("=" * 96)
    print("  현행: " + " · ".join(f"{d['name']} {s:.4f}" for d, s in zip(folds, base))
          + f"  → 평균 {b_avg:.4f}\n")

    # ---- 중심 m 은 폴드에 의존하지 않게 고정 상수로 둔다 (폴드마다 다르면 그것도 자유도) ----
    M = float(np.mean([np.mean(np.log1p(np.maximum(d["p"], 0))) for d in folds]))
    print(f"  중심 m = log1p 평균 = {M:.4f}  (원공간 {np.expm1(M):.2f})\n")

    def expand(p, k, m=M):
        return np.expm1(m + k * (np.log1p(np.maximum(p, 0)) - m))

    print("=" * 96)
    print("① 펴기 계수 k 스윕   log1p(p') = m + k·(log1p(p) − m)")
    print("=" * 96)
    KS = np.round(np.arange(0.90, 1.351, 0.01), 2)
    curve = {}
    print(f"  {'k':>6s} " + "".join(f"{d['name']:>11s}" for d in folds)
          + f"{'평균':>10s}{'현행대비':>10s}{'일관':>6s}")
    for k in KS:
        s = [sc(d, expand(d["p"], k)) for d in folds]
        curve[float(k)] = s
        if abs(k * 100 - round(k * 100)) < 1e-9 and round(k * 100) % 2 == 0:
            both = all(s[i] < base[i] for i in range(4))
            g = b_avg - np.mean(s)
            mark = "  ←현행" if abs(k - 1.0) < 1e-9 else ""
            print(f"  {k:>6.2f} " + "".join(f"{x:>11.4f}" for x in s)
                  + f"{np.mean(s):>10.4f}{g:>+10.4f}{'○' if both else '×':>5s}{mark}")
    best_each = [KS[int(np.argmin([curve[float(k)][i] for k in KS]))] for i in range(4)]
    kb = KS[int(np.argmin([np.mean(curve[float(k)]) for k in KS]))]
    gain = b_avg - float(np.mean(curve[float(kb)]))
    allb = all(curve[float(kb)][i] < base[i] for i in range(4))
    print(f"\n  폴드별 최적 k: " +
          " · ".join(f"{d['name']} {b:.2f}" for d, b in zip(folds, best_each)))
    print(f"  평균 기준 최적 k = {kb:.2f}   개선 {gain:+.4f}   "
          f"4폴드 일관 {'○' if allb else '×'}   (문턱 2σ={SIGMA2})")

    # ---- 교차검증: 두 폴드에서 k를 맞추고 나머지 두 폴드에서 채점 ----
    print("\n" + "=" * 96)
    print("② 정직성 검사 — 한쪽에서 k 를 맞추고 **다른 쪽에서 채점**")
    print("=" * 96)
    honest = []
    for fit_idx, test_idx, lab in [((0, 1), (2, 3), "가까움→멂"),
                                   ((2, 3), (0, 1), "멂→가까움"),
                                   ((0, 3), (1, 2), "겨울→봄"),
                                   ((1, 2), (0, 3), "봄→겨울")]:
        kf = KS[int(np.argmin([np.mean([curve[float(k)][i] for i in fit_idx])
                               for k in KS]))]
        s_new = float(np.mean([curve[float(kf)][i] for i in test_idx]))
        s_old = float(np.mean([base[i] for i in test_idx]))
        honest.append(dict(lab=lab, k=float(kf), new=s_new, old=s_old,
                           gain=s_old - s_new))
        print(f"  {lab:<12s} 맞춘 k={kf:.2f}  →  채점 {s_old:.4f} → {s_new:.4f}   "
              f"개선 {s_old-s_new:+.4f}")
    hg = float(np.mean([h["gain"] for h in honest]))
    print(f"\n  교차 평균 개선 {hg:+.4f}   ← **이게 진짜 기대값** "
          f"({'2σ 초과' if hg > SIGMA2 else '2σ 미달'})")

    # ---- 3구간 버전: 자유도를 늘릴 값어치가 있는가 ----
    print("\n" + "=" * 96)
    print("③ 비교군 — 3구간 배율 (자유도 3). 자유도를 늘릴 값어치가 있는가")
    print("=" * 96)
    T1, T2 = 2.0, 10.0
    best3, bs3 = None, 9.9
    for c1 in np.arange(0.70, 1.06, 0.05):
        for c2 in np.arange(0.85, 1.16, 0.05):
            for c3 in np.arange(0.95, 1.31, 0.05):
                s = []
                for d in folds:
                    p = d["p"].copy()
                    q = np.where(p < T1, c1 * p, np.where(p < T2, c2 * p, c3 * p))
                    s.append(sc(d, q))
                if np.mean(s) < bs3:
                    bs3, best3 = float(np.mean(s)), (c1, c2, c3, s)
    c1, c2, c3, s3 = best3
    print(f"  경계 {T1}/{T2} · 최적 배율 ({c1:.2f}, {c2:.2f}, {c3:.2f})")
    print("  " + " · ".join(f"{d['name']} {x:.4f}" for d, x in zip(folds, s3))
          + f"  → 평균 {bs3:.4f}  개선 {b_avg-bs3:+.4f}  "
          f"일관 {'○' if all(s3[i] < base[i] for i in range(4)) else '×'}")
    print(f"  ※ 자유도 3은 4개 폴드에 맞춰 고른 값이라 낙관 편향이 있다. k(자유도 1)와 비교용.")

    # ---- 결론 ----
    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    ok = hg > SIGMA2 and allb
    print(f"  단순 스윕 최적 k={kb:.2f} 개선 {gain:+.4f} · 4폴드 일관 {'○' if allb else '×'}")
    print(f"  교차 검증 개선 {hg:+.4f}")
    if ok:
        print(f"  → ★ 채택 권고. 제출: python make_submission.py <이름> --expand {kb:.2f}")
    elif hg > 0 and allb:
        print(f"  → ⚠️ 방향은 일관되나 2σ 미달. 제출 1회로 확인할 값어치는 있다.")
    else:
        print(f"  → 기각. 현행 유지.")

    json.dump(dict(center=M, k_curve={str(k): v for k, v in curve.items()},
                   base=base, best_k=float(kb), gain=gain, consistent=bool(allb),
                   honest=honest, honest_gain=hg,
                   seg3=dict(t=[T1, T2], c=[float(c1), float(c2), float(c3)],
                             score=bs3, gain=b_avg - bs3)),
              open(os.path.join(C.EXPERIMENTS, "phase6b_expand.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase6b_expand.json")


if __name__ == "__main__":
    main()
