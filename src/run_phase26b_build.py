# -*- coding: utf-8 -*-
"""Phase 26-b — 계절 게이트 제출본 생성 + 사전등록. **제출은 사람이 한다.**

게이트 (자유도 = w 하나. 계절 경계는 검증구간이 정한 것이라 맞출 게 없다):
    창 끝 날짜가 우리 **봄 검증구간(2/23~6/8)** 안에 들어오는 TEST 파일에만 섞는다.
    나머지는 v24 그대로.

근거 (테스트 점수 0회 사용):
    학습량 고정 실험 + 기존 4폴드 = 총 7개 측정에서
      봄   k = 1.024 / 1.033 / 1.038  (학습행 156k~250k, ±0.007 안)  w* +0.38~+0.43
      겨울 k = 1.094 / 1.142 / 1.179  (불안정)                        w* −0.11~+0.20
      가을 k = 1.267                                                  w* −0.06
    ρ 는 0.77~0.88 로 계절 의존이 약하다. 갈리는 건 k 하나다.
    raw(스냅 없음) 실측 이득: 봄 +0.00664(w=0.20) · 겨울 −0.00283 · 가을 −0.00299

w = 0.20 을 쓴다. 봄 raw 최적은 0.25(+0.00677)이나 0.20(+0.00664)과 차이가 없고,
이 프로젝트는 최적점을 그대로 쓰면 과녁을 넘긴 전력이 있어 한 칸 축소한다.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config as C
import dataio as D

V24 = os.path.join(C.EXPERIMENTS, "phase18_prune3_raw.npy")
TM = os.path.join(C.EXPERIMENTS, "tm_ridge_only_test_raw.npy")
OUT = os.path.join(C.EXPERIMENTS, "phase26b_springgate_raw.npy")

W_SPRING = 0.20
SPRING = ((2, 23), (6, 8))          # 우리 봄 검증구간 (월, 일)


def in_spring(d):
    md = (d.month, d.day)
    return SPRING[0] <= md <= SPRING[1]


def frac(v):
    return 100 * (v < 1.8).mean()


def main():
    items = D.item_order()
    n = len(items)
    v24, tm = np.load(V24), np.load(TM)
    assert v24.shape == tm.shape == (C.N_TEST * C.HORIZON * n,)

    print("TEST 창 끝 날짜 → 게이트")
    w_cell = np.zeros_like(v24)
    picked = []
    for t in range(C.N_TEST):
        te = D.load_test(t)
        _, tdates = D.to_matrix(te, items)
        end = pd.Timestamp(tdates[-1])
        spring = in_spring(end)
        if spring:
            s = t * C.HORIZON * n
            w_cell[s:s + C.HORIZON * n] = W_SPRING
            picked.append(f"TEST_{t:02d}")
        print(f"  TEST_{t:02d}  창끝 {end.date()}  "
              f"{'🌱 봄 → w=' + str(W_SPRING) if spring else '· w=0'}")

    blend = (1 - w_cell) * v24 + w_cell * tm
    np.save(OUT, blend)
    print(f"\n섞은 파일 {len(picked)}개: {picked}")
    print(f"바뀐 칸 {100*(np.abs(blend-v24) > 1e-9).mean():.1f}% "
          f"(= {len(picked)}/10 파일)")

    print(f"\n보정 정합 — raw<1.8 비율")
    print(f"  v24 {frac(v24):.2f}%  ·  게이트본 {frac(blend):.2f}%  "
          f"({frac(blend)-frac(v24):+.2f}%p {'✅' if abs(frac(blend)-frac(v24))<=3 else '❌'})")

    print("\n" + "=" * 78)
    print("📌 사전등록 (결과 보기 전)")
    print(f"""
  비교 대상 : v24_prune3  Private 0.4375952
  구성      : T07·T08·T09 에만 0.80·v24 + 0.20·팀원Ridge · 나머지 v24 · v17 후처리

  기대치 계산
    봄 raw 이득 +0.00664 × 봄 파일 비중 3/10 = **+0.0020** (스냅 미포함)
    드리프트 할인 : 내부 봄 폴드는 학습 후 0~3.5개월, TEST 봄은 9~12개월.
                    1탄 전이율 0.63 을 적용하면 +0.0013
  → 예상 Private **0.4356 ~ 0.4364**   (1등 0.437 아래)

  판정
    · Private < 0.4375952  → 계절 게이트 성립. 앙상블 축이 '조건부로' 열린다.
    · Private ≥ 0.4375952  → 봄 k=1.03 이 3회 재현되고도 전이 안 됨.
                             축 영구 종료 + 규칙 ⑮(폴드는 TEST 창을 대표 못함)의 최종 증거.

  ⚠️ 오염 고지 : Public={{T00,T01,T05,T07,T08}} 이 개선됐다는 걸 이미 알고 있고,
     게이트가 지목한 3개 중 T07·T08 이 거기 속한다. 게이트 자체는 내부 측정만으로
     유도되지만, **완전한 블라인드는 아니다.** 성공해도 각주가 붙는다.
  ⚠️ 여름(T00·T01)은 측정 구간이 없어 w=0 으로 뒀다. 보수적 선택이나 미지수다.
  ⚠️ 게이트 밖 재심 4번째다. 앞선 셋(램프·v25·v26)은 전부 졌다.
""")
    print("=" * 78)

    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "make_submission.py"),
                    "v27_springgate", "--raw", OUT, "--seg", "0.55,0.90,1.02",
                    "--t", "1.8,10", "--snap", "geom"],
                   check=True, cwd=os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()
