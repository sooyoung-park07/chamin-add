# -*- coding: utf-8 -*-
"""Phase 6-c — 3구간 보정의 정직성 검사 + 격자 확장.

6-b 결과: 자유도 1짜리 로그확장(k)은 교차검증 개선 **+0.0000**으로 무효.
          그런데 3구간 배율은 **+0.0081 · 4폴드 전부 개선**이 나왔다.

두 가지를 확인해야 채택할 수 있다:
  ① 최적 c1 이 **격자 경계(0.70)에 붙었다** → 더 아래에 최적이 있을 수 있다. 범위를 넓힌다.
  ② 그 +0.0081 은 **4개 폴드에 맞춰 고른 값**이다(낙관 편향).
     → 두 폴드에서 맞추고 **나머지 두 폴드에서 채점**해야 진짜 기대값이 나온다.

②가 핵심이다. 지금까지 이 프로젝트가 반복해서 당한 게 정확히 이 함정이다.

구간 경계(2, 10)는 **고정한다.** 경계까지 맞추면 자유도가 5로 늘어 과적합 위험이 커진다.
"""
import os
import json
import itertools

import numpy as np

import config as C
import features as F
from metrics import competition_score, make_weights

OOF = os.path.join(C.EXPERIMENTS, "phase6a_oof.npz")
NAMES = ["F2 겨울", "F3 봄", "FAR-봄", "FAR-겨울"]
FLOOR = 1.0
SIGMA2 = 0.0032
T1, T2 = 2.0, 10.0                       # 고정

C1 = np.round(np.arange(0.45, 1.061, 0.05), 2)     # 경계에 붙었으므로 아래로 확장
C2 = np.round(np.arange(0.80, 1.161, 0.05), 2)
C3 = np.round(np.arange(0.90, 1.351, 0.05), 2)


def main():
    ctx = F.Context()
    z = np.load(OOF)
    folds = [dict(name=NAMES[i], p=z[f"p{i}"].astype(np.float64),
                  y=z[f"y{i}"], iids=z[f"i{i}"]) for i in range(4)]

    def sc(d, p):
        return competition_score(d["y"], np.maximum(p, FLOOR), d["iids"],
                                 ctx.store_of_item, make_weights(1.0), ctx.n)

    def apply(p, c):
        c1, c2, c3 = c
        return np.where(p < T1, c1 * p, np.where(p < T2, c2 * p, c3 * p))

    base = [sc(d, d["p"]) for d in folds]
    print("=" * 96)
    print(f"Phase 6-c — 3구간 보정 (경계 {T1}/{T2} 고정) · 정직성 검사")
    print("=" * 96)
    print("  현행: " + " · ".join(f"{d['name']} {s:.4f}" for d, s in zip(folds, base))
          + f"  → 평균 {np.mean(base):.4f}")
    print(f"  격자 c1 {C1[0]}~{C1[-1]} · c2 {C2[0]}~{C2[-1]} · c3 {C3[0]}~{C3[-1]}"
          f"  ({len(C1)*len(C2)*len(C3)}개 조합)\n")

    # 모든 조합 × 모든 폴드 점수를 한 번에 계산해 캐시
    combos = list(itertools.product(C1, C2, C3))
    table = np.zeros((len(combos), 4))
    for j, c in enumerate(combos):
        for i, d in enumerate(folds):
            table[j, i] = sc(d, apply(d["p"], c))

    # ---------------------------------------------------------- 낙관 버전
    jb = int(np.argmin(table.mean(1)))
    cb, sb = combos[jb], table[jb]
    print("=" * 96)
    print("① 4폴드 전부에 맞춘 최적 (낙관 편향 — 이 숫자를 믿으면 안 된다)")
    print("=" * 96)
    print(f"  배율 ({cb[0]:.2f}, {cb[1]:.2f}, {cb[2]:.2f})")
    print("  " + " · ".join(f"{d['name']} {x:.4f}" for d, x in zip(folds, sb))
          + f"  → 평균 {sb.mean():.4f}  개선 {np.mean(base)-sb.mean():+.4f}"
          + f"  일관 {'○' if all(sb[i] < base[i] for i in range(4)) else '×'}")
    edge = [cb[0] in (C1[0], C1[-1]), cb[1] in (C2[0], C2[-1]), cb[2] in (C3[0], C3[-1])]
    print(f"  격자 경계에 붙었나: c1 {edge[0]} · c2 {edge[1]} · c3 {edge[2]}"
          + ("   ⚠️ 붙었으면 범위를 더 넓혀야 한다" if any(edge) else "   ✅ 내부 최적"))

    # ---------------------------------------------------------- 정직 버전
    print("\n" + "=" * 96)
    print("② ★ 정직성 검사 — 두 폴드에서 맞추고 **나머지 두 폴드에서 채점**")
    print("=" * 96)
    SPLITS = [((0, 1), (2, 3), "가까움 → 멂"),
              ((2, 3), (0, 1), "멂 → 가까움"),
              ((0, 3), (1, 2), "겨울 → 봄"),
              ((1, 2), (0, 3), "봄 → 겨울"),
              ((0, 2), (1, 3), "F2·FAR봄 → F3·FAR겨울"),
              ((1, 3), (0, 2), "F3·FAR겨울 → F2·FAR봄")]
    honest = []
    print(f"  {'분할':<24s}{'맞춘 배율':>20s}{'현행':>9s}{'보정후':>9s}{'개선':>9s}")
    for fit_i, test_i, lab in SPLITS:
        jf = int(np.argmin(table[:, list(fit_i)].mean(1)))
        cf = combos[jf]
        new = float(table[jf, list(test_i)].mean())
        old = float(np.mean([base[i] for i in test_i]))
        honest.append(dict(lab=lab, c=[float(x) for x in cf],
                           old=old, new=new, gain=old - new))
        print(f"  {lab:<24s}{f'({cf[0]:.2f},{cf[1]:.2f},{cf[2]:.2f})':>20s}"
              f"{old:>9.4f}{new:>9.4f}{old-new:>+9.4f}")

    hg = float(np.mean([h["gain"] for h in honest]))
    hpos = sum(1 for h in honest if h["gain"] > 0)
    print(f"\n  **교차 평균 개선 {hg:+.4f}** · {hpos}/6 분할에서 개선")
    print(f"  (낙관 버전 {np.mean(base)-sb.mean():+.4f} 와의 차이 "
          f"{abs((np.mean(base)-sb.mean()) - hg):.4f} 가 곧 과적합분)")

    # ---------------------------------------------------------- 안정적 배율
    med = [float(np.median([h["c"][k] for h in honest])) for k in range(3)]
    jm = int(np.argmin([abs(combos[j][0]-med[0]) + abs(combos[j][1]-med[1])
                        + abs(combos[j][2]-med[2]) for j in range(len(combos))]))
    sm = table[jm]
    print(f"\n  6개 분할이 고른 배율의 중앙값 = ({med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f})")
    print("  이 배율의 4폴드 점수: "
          + " · ".join(f"{d['name']} {x:.4f}" for d, x in zip(folds, sm))
          + f"  평균 {sm.mean():.4f}  개선 {np.mean(base)-sm.mean():+.4f}"
          + f"  일관 {'○' if all(sm[i] < base[i] for i in range(4)) else '×'}")

    print("\n" + "=" * 96)
    print("판정")
    print("=" * 96)
    adopt = hg > SIGMA2 and hpos >= 5
    print(f"  교차 평균 개선 {hg:+.4f} · 개선 분할 {hpos}/6 · 문턱 2σ={SIGMA2}")
    if adopt:
        print(f"  → ★ 채택 권고. 배율 ({med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}), 경계 {T1}/{T2}")
    elif hg > 0 and hpos >= 4:
        print(f"  → ⚠️ 방향 일관하나 2σ 미달. **제출 1회로 확인할 값어치 있음.**")
    else:
        print(f"  → 기각. 현행 유지.")

    json.dump(dict(t=[T1, T2], grid_best=[float(x) for x in cb],
                   grid_best_scores=[float(x) for x in sb],
                   honest=honest, honest_gain=hg, honest_pos=hpos,
                   median_c=med, median_scores=[float(x) for x in sm],
                   base=base),
              open(os.path.join(C.EXPERIMENTS, "phase6c_seg3.json"),
                   "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n저장: experiments/phase6c_seg3.json")


if __name__ == "__main__":
    main()
