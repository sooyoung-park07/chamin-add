# -*- coding: utf-8 -*-
"""Phase 21-g — 제출 후보 2건 생성 + 사전등록. **제출은 사람이 한다.**

후보
  S1  tm_v17    : 팀원 Ridge raw + 우리 v17 후처리        (진단용)
                  → 그들 0.4957 중 얼마가 '모델'이고 얼마가 '후처리 부재'였나를 가른다.
                    부수 효과로 전체 학습량에서의 σ비 k 를 알게 되어 w* 를 정밀화할 수 있다.
  S2  v26_b10   : 0.90·v24_raw + 0.10·팀원Ridge_raw + v17 후처리   (본 후보)

섞는 규약 (21-c 와 동일)
  · raw 끼리 섞는다. 후처리(3구간 배율 → 하한 → 정수 스냅)는 **섞은 뒤 한 번만.**
  · 팀원 전역 ×0.94 는 벗긴 상태. 우리 3구간 배율이 그 역할을 한다.
  · 그들 blend(0.68/0.32) 가 아니라 **ridge 단독**을 쓴다 — baseline 32% 는
    우리 `d_posmean` 피처와 겹쳐 잔차 상관을 0.86→0.91 로 올린다(21-d).
"""
import os
import subprocess
import sys

import numpy as np

import config as C

V24 = os.path.join(C.EXPERIMENTS, "phase18_prune3_raw.npy")
TM = os.path.join(C.EXPERIMENTS, "tm_ridge_only_test_raw.npy")
W = 0.10
OUT = os.path.join(C.EXPERIMENTS, f"phase21g_blend{int(W*100):02d}_raw.npy")


def frac(v):
    return (100 * (v < 1.8).mean(), 100 * ((v >= 1.8) & (v < 10)).mean(),
            100 * (v >= 10).mean())


def main():
    v24 = np.load(V24)
    tm = np.load(TM)
    assert v24.shape == tm.shape, "길이 불일치"
    blend = (1 - W) * v24 + W * tm
    np.save(OUT, blend)

    print("=" * 78)
    print("보정 정합 — 우리 3구간 배율이 그대로 유효한가 (v24 대비 3%p 이내가 기준)")
    print(f"{'벡터':<16s}{'raw<1.8':>9s}{'1.8~10':>9s}{'>=10':>8s}{'중앙값':>9s}")
    for lab, v in (("v24 (기준)", v24), ("팀원 ridge", tm), (f"blend w={W}", blend)):
        a, b, c = frac(v)
        print(f"{lab:<16s}{a:>8.2f}%{b:>8.2f}%{c:>7.2f}%{np.median(v):>9.3f}")
    d = frac(blend)[0] - frac(v24)[0]
    print(f"  → blend 는 v24 대비 {d:+.2f}%p "
          f"{'✅ 통과' if abs(d) <= 3 else '❌ 게이트 밖'}")
    print(f"  ※ 팀원 단독은 {frac(tm)[0]-frac(v24)[0]:+.2f}%p 로 게이트 밖이다. "
          f"S1 은 진단용이라 그대로 간다.")

    print("\n" + "=" * 78)
    print("두 예측이 실제로 갈리는 정도")
    ch = 100 * (np.abs(blend - v24) > 1e-9).mean()
    print(f"  raw 가 바뀐 칸 {ch:.1f}%  ·  로그공간 평균 이동 "
          f"{np.mean(np.log1p(blend) - np.log1p(v24)):+.5f}")

    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    for stamp, raw in ((f"tm_v17", TM), (f"v26_b{int(W*100):02d}", OUT)):
        print()
        subprocess.run([py, os.path.join(here, "make_submission.py"), stamp,
                        "--raw", raw, "--seg", "0.55,0.90,1.02",
                        "--t", "1.8,10", "--snap", "geom"], check=True, cwd=here)


if __name__ == "__main__":
    main()
