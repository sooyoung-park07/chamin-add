# -*- coding: utf-8 -*-
"""Phase 12-c — 램프 부스트 최종 검증 + 제출본 v21/v22 생성.

확정된 규칙 (자유도: m 하나. 경계 2.0 과 제외조건은 이론·독립구간에서 유도)
    r = (창 last7 평균 + 1) / (창 prev7 평균 + 1)
    r > 2  AND  창 last7 의 공휴일 수 < 2   →   base × 1.5  후 기하 스냅

근거 사슬
    · Q7: r>2 에서 부호편향 −0.30~−0.64 (과소예측). 사후 최적 배율 겨울 1.58 / 봄 1.53.
    · m 스윕 1.2~1.7 이 4폴드 전부 양수, 1.5~1.7 고원 (규칙 7 통과).
    · 양방향 전이: 겨울→봄 +0.0082 · 봄→겨울 +0.0057 (규칙 3 통과).
    · 제3 독립구간(2023 추석, 학습기간)으로 두 체제 확인:
      명절 급등 keep 0.59~0.78 (꺼짐) vs 계절 램프 keep 1.15~1.19 (지속).
      → 연휴 제외조건. 공휴일 달력은 도메인 지식으로 허용 (대회 규칙).

v21 = 연휴 제외 부스트 (본 후보) · v22 = 무조건 부스트 (기전 분리용 진단 팔)
"""
import os
import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

M = 1.5
RAW = os.path.join(C.EXPERIMENTS, "phase10_ens83_raw.npy")   # v17 의 원본

FOLDS = [
    ("F2 겨울",  "2023-11-24", "2023-11-24", "2024-02-22"),
    ("F3 봄",    "2024-02-23", "2024-02-23", "2024-06-08"),
    ("FAR-봄",   "2023-11-24", "2024-02-23", "2024-06-08"),
    ("FAR-겨울", "2023-08-25", "2023-11-24", "2024-02-22"),
]


def seg_base(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    return np.maximum(p, 1.0)


def gsnap(p):
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a = np.abs(a); p = np.abs(p)
    den = a + p
    out = np.zeros(len(a)); m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def main():
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    didx = pd.DatetimeIndex(dates)
    nd = mat.shape[1]
    ctx.set_proxy(F.pick_proxy_items(mat, dates, nd, ctx.store_codes))
    hols = set(pd.to_datetime(list(C.HOLIDAYS)))
    hol_arr = np.array([d in hols for d in didx])
    d20 = np.load(os.path.join(C.EXPERIMENTS, "phase10e_oof20.npz"))

    # ── 공휴일 달력 커버리지 확인 (테스트 기간 2024-06~2025-05) ─────
    hs = sorted(h for h in hols if pd.Timestamp("2024-06-01") <= h <= pd.Timestamp("2025-06-01"))
    print("공휴일 달력 (테스트 기간):", ", ".join(str(h.date()) for h in hs) or "⚠️ 없음!")
    chuseok24 = [h for h in hs if pd.Timestamp("2024-09-14") <= h <= pd.Timestamp("2024-09-19")]
    print(f"  추석 2024 포함 여부: {len(chuseok24)}일 → "
          + ("OK" if len(chuseok24) >= 2 else "⚠️ 달력에 추석이 없다 — 제외규칙 작동 불가"))

    # ── 내부 4폴드: 제외규칙 있는/없는 버전 ─────────────────────────
    print("\n" + "=" * 92)
    print(f"내부 검증 — m={M} · r>2 부스트 (v22=무조건 / v21=연휴 제외)")
    print("=" * 92)
    print(f"  {'폴드':<10s}{'v17 점수':>10s}{'v22 이득':>10s}{'v21 이득':>10s}{'제외된 칸':>10s}")
    for fi, (name, cut, v0, v1) in enumerate(FOLDS):
        va = V.origins(dates, v0, v1, nd)
        _, yva, mva = F.build_samples(mat, dates, va, ctx)
        assert np.array_equal(yva, d20[f"y{fi}"])
        raw = d20[f"ps{fi}"].mean(0).astype(np.float64)
        W, _ = cell_weights(yva, mva[:, 2], ctx.store_of_item, ctx.n)
        r = np.empty(len(yva)); h7 = np.empty(len(yva), dtype=int)
        for o in np.unique(mva[:, 0]):
            m = mva[:, 0] == o
            it = mva[m, 2]
            l7 = mat[:, o - 6:o + 1].mean(1); p7 = mat[:, o - 13:o - 6].mean(1)
            r[m] = (l7[it] + 1.0) / (p7[it] + 1.0)
            h7[m] = int(hol_arr[o - 6:o + 1].sum())
        base = seg_base(raw); ours = gsnap(base)
        cur = float((W * loss(yva, ours)).sum())
        b22 = r > 2.0
        b21 = b22 & (h7 < 2)
        out = []
        for b in (b22, b21):
            p = base.copy(); p[b] = M * p[b]
            out.append(float((W * (loss(yva, ours) - loss(yva, gsnap(p)))).sum()))
        print(f"  {name:<10s}{cur:>10.5f}{out[0]:>+10.5f}{out[1]:>+10.5f}"
              f"{(b22 & ~b21).sum():>9,}")

    # ── 제출본 생성 ─────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("제출본 생성 (원본 = v17 앙상블 raw)")
    print("=" * 92)
    raw = np.load(RAW)
    n = ctx.n
    assert raw.shape[0] == C.N_TEST * C.HORIZON * n
    mask22 = np.zeros(raw.shape[0], dtype=bool)
    mask21 = np.zeros(raw.shape[0], dtype=bool)
    for t in range(C.N_TEST):
        te = D.load_test(t)
        tmat, tdates = D.to_matrix(te, ctx.items)
        tdi = pd.DatetimeIndex(tdates)
        o = tmat.shape[1] - 1
        l7 = tmat[:, o - 6:o + 1].mean(1); p7 = tmat[:, o - 13:o - 6].mean(1)
        rr = (l7 + 1.0) / (p7 + 1.0)
        nhol = int(sum(d in hols for d in tdi[-7:]))
        b = rr > 2.0
        for h in range(C.HORIZON):
            off = t * C.HORIZON * n + h * n
            mask22[off:off + n] = b
            if nhol < 2:
                mask21[off:off + n] = b
        ex = "  (연휴 " + str(nhol) + "일 → v21 제외)" if nhol >= 2 else ""
        print(f"  TEST_{t:02d}  r>2 품목 {b.sum():>3d}개{ex}")

    base = seg_base(raw)
    tpl = D.load_submission_template()
    items = D.item_order()
    rk = tpl.columns[0]
    for stamp, mask in (("v21_ramp_holex", mask21), ("v22_ramp_all", mask22)):
        p = base.copy(); p[mask] = M * p[mask]
        p = gsnap(p)
        out = tpl.copy(); out[items] = out[items].astype(float)
        off = 0
        for t in range(C.N_TEST):
            blk = p[off:off + C.HORIZON * n].reshape(C.HORIZON, n)
            off += C.HORIZON * n
            for h in range(1, C.HORIZON + 1):
                out.loc[out.index[out[rk] == f"TEST_{t:02d}+{h}일"][0], items] = blk[h - 1]
        out[items] = out[items].round(2)
        path = os.path.join(C.SUBMISSIONS, f"submission_{stamp}.csv")
        out.to_csv(path, index=False, encoding="utf-8-sig")
        v17 = gsnap(base)
        print(f"\n  저장 submission_{stamp}.csv — 부스트 칸 {mask.sum():,} "
              f"({100 * mask.mean():.1f}%) · v17 과 다른 칸 {100 * (p != v17).mean():.1f}%"
              f" · min {p.min():.0f} / max {p.max():.0f}")


if __name__ == "__main__":
    main()
