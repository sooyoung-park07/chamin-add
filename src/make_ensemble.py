# -*- coding: utf-8 -*-
"""제출본 CSV 두 개를 등가중 평균해 '원본 예측 .npy' 로 되돌린다. **재학습 없음.**

    python make_ensemble.py <출력이름> <csv1> <csv2> [...]

왜 이게 필요한가
    Phase 5-d 의 앙상블 시험은 v4(최악) 를 섞은 결함 실험이었고, 제출한 v5 도 마찬가지였다.
    **v8(합산 1위 베이스) + v3(XGBoost)** 조합은 한 번도 안 해봤다.
    두 CSV 가 남아 있으므로 **재학습이 0** 이다 → 시드 노이즈도 0 → 문턱 없이 판정 가능.

규약
    · 등가중만. 가중치 최적화는 Phase 5-a 에서 이미 기각(최적가중 <= 등가중).
    · 하한은 여기서 걸지 않는다. make_submission.py 가 배율 뒤에 한 번만 건다.
      (입력 CSV 는 이미 하한 1.0 이 걸린 상태라 완전한 pre-floor 는 아니다 — 아래 진단 참고)

덤으로 하는 일 — **매핑 검증 (제출 0회)**
    v8 CSV 는 `max(phase6a_test_raw.npy, 1.0).round(2)` 와 정확히 같아야 한다.
    일치하면 `.npy -> 제출 템플릿` 매핑이 옳다는 뜻이라, 이걸 확인하려고
    제출을 한 장 쓸 필요가 없어진다.
"""
import os
import sys

import numpy as np
import pandas as pd

import config as C
import dataio as D

RAW = os.path.join(C.EXPERIMENTS, "phase6a_test_raw.npy")


def csv_to_flat(path, items, rk):
    """제출 CSV -> phase6a_test_raw.npy 와 같은 평탄 순서의 1차원 배열."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert list(df.columns[1:]) == items, f"열 순서 불일치: {os.path.basename(path)}"
    n = len(items)
    flat = np.empty(C.N_TEST * C.HORIZON * n, dtype=np.float64)
    off = 0
    for t in range(C.N_TEST):
        for h in range(1, C.HORIZON + 1):
            key = f"TEST_{t:02d}+{h}일"
            row = df.loc[df[df.columns[0]] == key, items]
            assert len(row) == 1, f"행 없음/중복: {key}"
            flat[off:off + n] = row.values[0].astype(float)
            off += n
    return flat


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        print(__doc__)
        return
    stamp, paths = argv[0], argv[1:]

    items = D.item_order()
    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    assert list(tpl.columns[1:]) == items, "제출 템플릿 열 순서 불일치"

    flats = []
    for p in paths:
        f = p if os.path.isabs(p) else os.path.join(C.SUBMISSIONS, p)
        flats.append(csv_to_flat(f, items, rk))
        print(f"  읽음: {os.path.basename(f)}  "
              f"min {flats[-1].min():.2f} / 중앙 {np.median(flats[-1]):.2f} / "
              f"max {flats[-1].max():.2f}")

    # ── 매핑 검증 (제출 0회) ─────────────────────────────────────────
    raw = np.load(RAW)
    v8_expect = np.round(np.maximum(raw, 1.0), 2)
    for p, f in zip(paths, flats):
        if "v8" in os.path.basename(p):
            same = np.allclose(f, v8_expect, atol=1e-9)
            md = float(np.abs(f - v8_expect).max())
            print(f"\n  [매핑 검증] {os.path.basename(p)} vs max(raw,1).round(2): "
                  f"{'일치' if same else '불일치'} (최대차 {md:.6f})")
            print("    -> .npy → 제출 템플릿 매핑이 옳다는 뜻. "
                  "휴면칸 검증에 제출을 쓸 필요 없음." if same else
                  "    -> ⚠️ 매핑 또는 원본이 어긋난다. 원인 규명 전까지 판정 금지.")

    # ── 등가중 평균 ──────────────────────────────────────────────────
    ens = np.mean(flats, axis=0)

    # 잔차 상관 (다양성이 남아 있는지)
    if len(flats) == 2:
        a, b = np.log1p(flats[0]), np.log1p(flats[1])
        print(f"\n  두 예측의 로그공간 상관 = {np.corrcoef(a, b)[0,1]:.4f}"
              f" · 평균절대차 {np.abs(flats[0]-flats[1]).mean():.2f}")

    # ── v9 보정 정합 진단 (θ = 1/0.55 = 1.818 아래 칸 비율) ──────────
    th = 1.0 / 0.55
    print(f"\n  [보정 정합 진단] raw < {th:.3f} 비율")
    print(f"    v8 원본  {100*(raw < th).mean():.2f}%")
    print(f"    앙상블   {100*(ens < th).mean():.2f}%"
          f"   (차이 {100*((ens < th).mean() - (raw < th).mean()):+.2f}%p)")
    gap = abs((ens < th).mean() - (raw < th).mean()) * 100
    print("    -> 3%p 이내: v9 보정을 그대로 얹어 비교해도 된다." if gap < 3 else
          "    -> ⚠️ 3%p 초과: 모델 효과와 보정 부정합이 섞인다. 판정 보류.")

    out = os.path.join(C.EXPERIMENTS, f"{stamp}_raw.npy")
    np.save(out, ens)
    print(f"\n저장: {os.path.basename(out)}  ({len(flats)}개 등가중)")
    print(f"  min {ens.min():.2f} / 중앙 {np.median(ens):.2f} / "
          f"평균 {ens.mean():.2f} / max {ens.max():.2f}")
    print(f"\n다음: python make_submission.py <이름> --raw {out} --seg 0.55,0.90,1.02")


if __name__ == "__main__":
    main()
