# -*- coding: utf-8 -*-
"""Phase 22 — 혼합이 이득인 창과 손해인 창을 **미리** 구분할 수 있는가.

동기. 실측에서 혼합(w=0.10)은 TEST 10개 중 5개에서 −0.0016 이득, 나머지 5개에서
+0.0033 손해였다. **효과의 부호가 창마다 뒤집힌다.** 그렇다면 전역 스칼라 w 대신
"이 창에서는 섞고 저 창에서는 안 섞는다"는 라우팅이 가능한가?

가능하려면 조건이 하나 있다: **채점 결과를 보지 않고, 28일 창과 달력만으로**
그 부호를 예측할 수 있어야 한다. 채점 결과로 고르면 시험지 맞춤이다(Phase 14 에서 폐기).

그래서 여기서 재는 것:
  ① 내부 폴드에서도 창마다 부호가 갈리는가 (실측 현상이 재현되는가)
  ② 갈린다면, 창에서 **미리 계산 가능한 서술자**가 그 부호를 예측하는가
  ③ 예측한다면 그 관계가 **폴드 4개에서 같은 방향**인가 (규칙 ③·⑧)

서술자는 전부 그 창의 28일 + 달력에서만 나온다 (대회 규칙 안전).
표본이 폴드당 origin 13~16개, 합쳐 ~58개뿐이라 **상관 하나로 결론 내지 않는다.**
부호 일관성을 본다.
"""
import os

import numpy as np
import pandas as pd

import config as C
import dataio as D
import features as F
import validate as V
from run_phase10c_thresholds import cell_weights

W_BLEND = 0.10
FOLDS = [("F2 겨울", "2023-11-24", "2024-02-22"),
         ("F3 봄", "2024-02-23", "2024-06-08"),
         ("FAR-봄", "2024-02-23", "2024-06-08"),
         ("FAR-겨울", "2023-11-24", "2024-02-22")]
NPZ = os.path.join(C.EXPERIMENTS, "phase21c_oof.npz")


def seg_snap(raw):
    p = np.where(raw < 1.8, 0.55 * raw, np.where(raw < 10.0, 0.90 * raw, 1.02 * raw))
    p = np.maximum(p, 1.0)
    k = np.maximum(np.floor(p), 1.0)
    return np.maximum(np.where(p >= np.sqrt(k * (k + 1.0)), k + 1.0, k), 1.0)


def loss(a, p):
    a, p = np.abs(a), np.abs(p)
    den = a + p
    out = np.zeros(len(a))
    m = den > 0
    out[m] = 2.0 * np.abs(a[m] - p[m]) / den[m]
    return out


def descriptors(mat, dates, o, store_codes, n_store):
    """origin o 의 28일 창 + 예측 주간 달력에서만 뽑는 서술자. 채점 정보 없음."""
    win = np.maximum(mat[:, o - C.WINDOW + 1: o + 1].astype(float), 0.0)
    wd = pd.DatetimeIndex(dates[o - C.WINDOW + 1: o + 1])
    fut = pd.DatetimeIndex([dates[o] + pd.Timedelta(days=h)
                            for h in range(1, C.HORIZON + 1)])

    store = np.zeros((n_store, C.WINDOW))
    np.add.at(store, store_codes, win)
    total = store.sum(0)                       # 리조트 일별 총합
    last7, prev7 = total[-7:].mean(), total[-14:-7].mean()

    hol = set(pd.to_datetime(C.HOLIDAYS))
    d = {}
    d["level"] = np.log1p(total.mean())
    d["ramp"] = (last7 + 1.0) / (prev7 + 1.0)          # 창 안 추세 (급등/급락)
    d["cv"] = total.std() / max(total.mean(), 1e-9)     # 창 안 변동성
    d["closed"] = 1.0 - (store > 0).mean()              # 영업장×일 휴점 비율
    d["zero"] = (win == 0).mean()                       # 품목×일 0 비율
    d["nz_store"] = (store.sum(1) > 0).mean()           # 창 안 영업한 영업장 비율
    # 체제 전환 지표 — 창 안에서 영업 영업장 수가 바뀌었는가
    open_first = (store[:, :14].sum(1) > 0).sum()
    open_last = (store[:, 14:].sum(1) > 0).sum()
    d["regime_shift"] = abs(int(open_last) - int(open_first))
    d["month"] = float(fut[0].month)
    d["holidays"] = float(sum(t.normalize() in hol for t in fut))
    d["weekend"] = float(sum(t.dayofweek >= 5 for t in fut))
    # 시즌 경계까지의 부호 있는 거리 (팀원 모델이 가진 피처 계열)
    t0 = fut[0]
    for nm, mth, day in (("d_hwadam_open", 3, 29), ("d_hwadam_close", 11, 30),
                         ("d_ski_close", 3, 5)):
        cand = [pd.Timestamp(year=y, month=mth, day=day)
                for y in (t0.year - 1, t0.year, t0.year + 1)]
        d[nm] = float(min(((c - t0).days for c in cand), key=abs))
    d["near_boundary"] = float(min(abs(d["d_hwadam_open"]), abs(d["d_hwadam_close"]),
                                   abs(d["d_ski_close"])))
    return d


def main():
    z = np.load(NPZ)
    ctx = F.Context()
    tr = D.load_train()
    mat, dates = D.to_matrix(tr, ctx.items)
    nd = mat.shape[1]
    n_item = mat.shape[0]
    sc_codes = ctx.store_codes
    n_store = int(np.max(sc_codes)) + 1

    rows = []
    print("=" * 96)
    print(f"① 창마다 부호가 갈리는가 (혼합 w={W_BLEND} · +가 이득)")
    for fname, v0, v1 in FOLDS:
        va = V.origins(dates, v0, v1, nd)
        y = z[f"y|{fname}"]
        ours = z[f"ours|{fname}"]
        tm = z[f"tm_ridge|{fname}"]
        iid = z[f"iid|{fname}"].astype(int)
        blk = C.HORIZON * n_item
        assert len(y) == len(va) * blk, f"{fname} 행수 불일치 {len(y)} vs {len(va)*blk}"

        W, _ = cell_weights(y, iid, ctx.store_of_item, ctx.n)
        la = W * loss(y, seg_snap(ours))
        lb = W * loss(y, seg_snap((1 - W_BLEND) * ours + W_BLEND * tm))
        gains = np.array([la[i * blk:(i + 1) * blk].sum() - lb[i * blk:(i + 1) * blk].sum()
                          for i in range(len(va))])
        pos = int((gains > 0).sum())
        print(f"  [{fname:<8s}] origin {len(va):>2d}개 · 합계 {gains.sum():+.5f} · "
              f"이득 창 {pos}/{len(va)} · 창별 표준편차 {gains.std():.5f} · "
              f"최선 {gains.max():+.5f} / 최악 {gains.min():+.5f}")
        for i, o in enumerate(va):
            rows.append(dict(fold=fname, origin=o, date=dates[o], gain=gains[i],
                             **descriptors(mat, dates, o, sc_codes, n_store)))

    df = pd.DataFrame(rows)
    keys = [c for c in df.columns if c not in ("fold", "origin", "date", "gain")]
    print(f"\n  전체 {len(df)}개 창 · 이득 창 {(df.gain>0).sum()}/{len(df)} · "
          f"합계 {df.gain.sum():+.5f}")
    print("  → 창별 편차가 폴드 합계보다 크면, '평균 +0.0018'은 상쇄의 결과일 뿐이다.")

    print("\n" + "=" * 96)
    print("② 미리 계산 가능한 서술자가 그 부호를 예측하는가 (폴드별 스피어만 상관)")
    print(f"{'서술자':<16s}" + "".join(f"{f[0]:>12s}" for f in FOLDS)
          + f"{'전체':>10s}{'부호일관':>9s}")
    hits = []
    for k in keys:
        rs = []
        for fname, _, _ in FOLDS:
            s = df[df.fold == fname]
            if s[k].nunique() < 3:
                rs.append(np.nan)
                continue
            rs.append(float(s[[k, "gain"]].corr(method="spearman").iloc[0, 1]))
        allr = float(df[[k, "gain"]].corr(method="spearman").iloc[0, 1])
        ok = [r for r in rs if not np.isnan(r)]
        cons = (sum(r > 0 for r in ok) if np.mean(ok) > 0 else sum(r < 0 for r in ok))
        mark = "✅" if cons == len(ok) and len(ok) == 4 and abs(np.mean(ok)) > 0.3 else ""
        print(f"{k:<16s}" + "".join(f"{r:>12.3f}" if not np.isnan(r) else f"{'—':>12s}"
                                    for r in rs)
              + f"{allr:>10.3f}{cons:>6d}/{len(ok)} {mark}")
        if mark:
            hits.append((k, np.mean(ok)))

    print("\n" + "=" * 96)
    if hits:
        print("③ 4/4 같은 방향 + |상관|>0.3 인 서술자:")
        for k, r in sorted(hits, key=lambda x: -abs(x[1])):
            print(f"     {k}  (평균 상관 {r:+.3f})")
        print("   → 라우팅 규칙의 후보다. 단 창 58개짜리 상관이므로 규칙 ⑧ 적용:")
        print("     다른 설계에서 재현돼야 하고, 문턱을 데이터로 맞추면 자유도가 생긴다.")
    else:
        print("③ **4폴드에서 같은 방향으로 예측하는 서술자가 하나도 없다.**")
        print("   → 창을 미리 보고 '섞을지 말지'를 정할 근거가 없다.")
        print("     라우팅은 붙일 자리가 없고, 남는 건 채점 결과로 고르는 것뿐인데")
        print("     그건 시험지 맞춤이라 성적이 아니다. **조건부 결합 축도 닫힌다.**")
    print("=" * 96)

    out = os.path.join(C.EXPERIMENTS, "phase22_window_signal.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {os.path.basename(out)} (창 {len(df)}개 × 서술자 {len(keys)}개)")


if __name__ == "__main__":
    main()
