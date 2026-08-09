# -*- coding: utf-8 -*-
"""Phase 21-a — 팀원 Ridge 예측을 우리 좌표계로 옮기고, 두 예측이 얼마나 다른지 잰다.

입력: teammate_repo/outputs/final_prediction_log.csv (그들 스크립트가 남기는 칸별 로그)
출력: experiments/tm_ridge_test_raw.npy      = 0.68·ridge + 0.32·같은요일양수평균
      experiments/tm_ridge_only_test_raw.npy = ridge 단독
      ※ 둘 다 **전역 ×0.94 미적용**. 우리 3구간 배율이 그 역할을 대신하므로
        배율을 두 번 걸지 않기 위해 벗겨둔다.

여기서 재는 것 (제출 0회):
  ① 우리 좌표계 매핑이 맞는지 (제출 CSV 재구성 대조)
  ② 두 예측의 상관 — 단, **예측 상관은 부풀려진다**(규칙 ⑬). 여기선 참고용이고
     진짜 판정은 21-b 의 OOF 잔차 상관에서 한다.
  ③ 보정 정합 — `raw<1.8` 비율. v24 는 33.99% 였다. 크게 벗어나면 우리 후처리를
     그들 예측에 그대로 옮기는 전제가 깨진다.
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D

TM_LOG = os.path.join(C.ROOT, "teammate_repo", "outputs", "final_prediction_log.csv")
TM_SUB = os.path.join(C.ROOT, "teammate_repo", "outputs", "submission_final.csv")
OUT_BLEND = os.path.join(C.EXPERIMENTS, "tm_ridge_test_raw.npy")
OUT_RIDGE = os.path.join(C.EXPERIMENTS, "tm_ridge_only_test_raw.npy")
V24_RAW = os.path.join(C.EXPERIMENTS, "phase18_prune3_raw.npy")

RIDGE_W, BASE_W, TM_MULT = 0.68, 0.32, 0.94


def to_our_order(df, col, items):
    """(test, horizon, item_id) 롱포맷 → 우리 npy 순서 [test][horizon][item] 평탄화."""
    piv = df.pivot_table(index=["test", "horizon"], columns="item_id",
                         values=col, aggfunc="first")
    piv = piv.reindex(columns=items)
    out = np.empty(C.N_TEST * C.HORIZON * len(items))
    off = 0
    for t in range(C.N_TEST):
        for h in range(1, C.HORIZON + 1):
            row = piv.loc[(f"TEST_{t:02d}", h)].to_numpy(float)
            assert np.isfinite(row).all(), f"TEST_{t:02d}+{h} 결측"
            out[off:off + len(items)] = row
            off += len(items)
    return out


def main():
    items = D.item_order()
    n = len(items)
    df = pd.read_csv(TM_LOG, encoding="utf-8-sig")
    print(f"팀원 예측 로그 {len(df):,}행 · 기대 {C.N_TEST * C.HORIZON * n:,}행")

    ridge = to_our_order(df, "pred_ridge", items)
    base = to_our_order(df, "pred_baseline", items)
    final = to_our_order(df, "prediction", items)
    blend = RIDGE_W * ridge + BASE_W * base

    # ① 매핑 검증 — 벗긴 뒤 다시 씌우면 그들 제출본과 같아야 한다
    recon = np.maximum(TM_MULT * blend, 1.0)
    print(f"\n① 매핑 검증  |recon − 그들최종| 최대 {np.abs(recon - final).max():.2e}")

    sub = pd.read_csv(TM_SUB, encoding="utf-8-sig")
    tpl = D.load_submission_template()
    assert list(sub.columns) == list(tpl.columns), "제출 열 순서 불일치"
    from_csv = np.concatenate([
        sub.loc[sub[sub.columns[0]] == f"TEST_{t:02d}+{h}일", items].to_numpy(float)[0]
        for t in range(C.N_TEST) for h in range(1, C.HORIZON + 1)])
    print(f"   |CSV 재구성 − recon| 최대 {np.abs(from_csv - recon).max():.2e}  "
          f"→ {'✅ 좌표계 일치' if np.abs(from_csv - recon).max() < 1e-6 else '❌ 불일치'}")

    np.save(OUT_BLEND, blend)
    np.save(OUT_RIDGE, ridge)
    print(f"\n저장: {os.path.basename(OUT_BLEND)} · {os.path.basename(OUT_RIDGE)}")

    # ② 우리 v24 원본과 비교
    v24 = np.load(V24_RAW)
    assert v24.shape == blend.shape
    lo = lambda v: np.log1p(np.maximum(v, 0.0))
    print("\n② 예측 비교 (참고용 — 진짜 판정은 21-b 잔차 상관)")
    for lab, v in (("ridge 단독", ridge), ("blend(0.68/0.32)", blend),
                   ("같은요일 양수평균", base)):
        r_raw = np.corrcoef(v, v24)[0, 1]
        r_log = np.corrcoef(lo(v), lo(v24))[0, 1]
        print(f"   {lab:<18s} vs v24 : 원공간 r={r_raw:.4f} · 로그공간 r={r_log:.4f}"
              f" · 중앙값 {np.median(v):7.3f} (v24 {np.median(v24):.3f})")

    d = lo(blend) - lo(v24)
    print(f"\n   로그공간 차이: 평균 {d.mean():+.4f} · 표준편차 {d.std():.4f}")
    print(f"   → 팀원 예측이 우리보다 체계적으로 "
          f"{'높다' if d.mean() > 0 else '낮다'} (exp {np.exp(d.mean()):.3f}배)")

    # ③ 보정 정합
    print("\n③ 보정 정합 (우리 후처리를 그대로 옮길 수 있는가)")
    for lab, v in (("v24 (우리)", v24), ("팀원 blend", blend), ("팀원 ridge", ridge)):
        print(f"   {lab:<12s} raw<1.8 {100*(v < 1.8).mean():5.2f}%"
              f" · 1.8~10 {100*((v >= 1.8) & (v < 10)).mean():5.2f}%"
              f" · >=10 {100*(v >= 10).mean():5.2f}%")
    print("   ※ v24 기준 33.99%. 3%p 이내면 통과로 봐 왔다.")


if __name__ == "__main__":
    main()
