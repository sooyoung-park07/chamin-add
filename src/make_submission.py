# -*- coding: utf-8 -*-
"""저장된 원본 테스트 예측에 보정만 입혀 제출본을 만든다. **재학습 없음 (수 초).**

    python make_submission.py <이름> [옵션]

옵션
    --mult C           전역 배율 (기본 1.0)
    --seg C1,C2,C3     3구간 배율. p<T1 → C1 · T1<=p<T2 → C2 · p>=T2 → C3
    --t   T1,T2        구간 경계 (기본 2,10)
    --floor F          하한 (기본 1.0)
    --snap MODE        정수 격자 사영. geom | arith | none (기본 none)
    --snapmax M        스냅을 적용할 상한. 이 값 이하인 칸만 스냅 (기본 무한)
    --raw PATH         원본 예측 .npy 경로 (기본 experiments/phase6a_test_raw.npy)

예)
    python make_submission.py v9_seg3   --seg 0.55,0.90,1.02
    python make_submission.py v10_lowonly --seg 0.55,1.00,1.00
    python make_submission.py v11_m096  --mult 0.96
    python make_submission.py v12_snapgeom --seg 0.55,0.90,1.02 --snap geom

보정 순서: p → (배율 적용) → **max(·, 하한)** → (정수 스냅).
하한은 반드시 **배율 뒤 한 번만** 건다 — max 는 비선형이라 순서가 결과를 바꾼다.

━━ 정수 스냅 (자유 파라미터 0개) ━━
실측 A는 항상 정수다. 그리고 SMAPE 항 2|A-P|/(A+P) 를 비율 r=A/P 로 쓰면
`phi(r) = r/(1+r)^2` 이고 **phi(r) = phi(1/r)** — 즉 손실이 **로그비율에 대해 대칭**이다
(수치 확인: phi(0.2)=phi(5)=0.13889).

그 대칭성 때문에 실측 질량이 정수에 몰려 있으면 두 정수 사이 구간이 오목해지고
최적점이 항상 정수 위에 앉는다. 예) 실측이 1 아니면 2일 때 기대손실:
    P=1 -> 0.3333 · P=1.5 -> 0.3429 · P=2 -> 0.3333   (가운데가 더 나쁘다)

경계도 산술중점이 아니라 **기하중점 sqrt(k(k+1))** 이다. 1과 2 사이는 1.5가 아니라 1.414.
`geom` 이 그 경계를, `arith` 는 통상 반올림을 쓴다. 맞출 상수가 없으므로
원본 예측이 바뀌어도 이 보정은 그대로 유효하다 (재적합 불필요).

OOF 4폴드 실측 (v9 보정 뒤 적용 · 양수면 개선 · 시드5):
    geom  전구간  +0.00302  (4/4 폴드)      arith p<=3  +0.00149  (4/4 폴드)

원본 예측: `experiments/phase6a_test_raw.npy` (하한 적용 전 · 시드5 평균 · v8 설정).
"""
import os
import sys

import numpy as np
import pandas as pd

import config as C
import dataio as D

RAW = os.path.join(C.EXPERIMENTS, "phase6a_test_raw.npy")


def parse(argv):
    o = dict(mult=1.0, seg=None, t=(2.0, 10.0), floor=1.0,
             snap="none", snapmax=np.inf, raw=RAW)
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--mult":
            o["mult"] = float(argv[i + 1]); i += 2
        elif a == "--seg":
            o["seg"] = tuple(float(x) for x in argv[i + 1].split(",")); i += 2
        elif a == "--t":
            o["t"] = tuple(float(x) for x in argv[i + 1].split(",")); i += 2
        elif a == "--floor":
            o["floor"] = float(argv[i + 1]); i += 2
        elif a == "--snap":
            o["snap"] = argv[i + 1]; i += 2
        elif a == "--snapmax":
            o["snapmax"] = float(argv[i + 1]); i += 2
        elif a == "--raw":
            o["raw"] = argv[i + 1]; i += 2
        else:
            raise SystemExit(f"알 수 없는 옵션: {a}")
    if o["seg"] is not None and len(o["seg"]) != 3:
        raise SystemExit("--seg 는 값 3개여야 한다 (예: 0.55,0.90,1.02)")
    if o["snap"] not in ("none", "geom", "arith"):
        raise SystemExit("--snap 은 none | geom | arith 중 하나")
    return o


def calibrate(raw, o):
    if o["seg"] is None:
        return o["mult"] * raw
    c1, c2, c3 = o["seg"]
    t1, t2 = o["t"]
    return np.where(raw < t1, c1 * raw, np.where(raw < t2, c2 * raw, c3 * raw))


def snap_to_int(p, mode, maxv=np.inf, floor=1.0):
    """정수 격자 사영. 하한 **뒤에** 적용한다 (하한도 정수라 순서 무관하나 명시).

    geom  : 경계 sqrt(k(k+1)).  손실이 로그비율 대칭이라 이게 옳은 경계다.
    arith : 통상 반올림 (경계 k+0.5). 대조군.
    """
    if mode == "none":
        return p
    q = np.asarray(p, dtype=np.float64).copy()
    m = q <= maxv
    if not m.any():
        return q
    if mode == "arith":
        q[m] = np.round(q[m])
    else:
        k = np.maximum(np.floor(q[m]), 1.0)
        q[m] = np.where(q[m] >= np.sqrt(k * (k + 1.0)), k + 1.0, k)
    return np.maximum(q, floor)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return
    stamp, o = argv[0], parse(argv[1:])

    items = D.item_order()
    tpl = D.load_submission_template()
    rk = tpl.columns[0]
    assert list(tpl.columns[1:]) == items, "제출 템플릿 열 순서 불일치"

    raw = np.load(o["raw"])
    n = len(items)
    assert raw.shape[0] == C.N_TEST * C.HORIZON * n, \
        f"원본 예측 길이 이상: {raw.shape[0]}"
    cur = np.maximum(raw, 1.0)                       # 비교 기준(= v8 제출본)
    p = np.maximum(calibrate(raw, o), o["floor"])    # 하한은 배율 뒤 한 번만
    pre_snap = p.copy()
    p = snap_to_int(p, o["snap"], o["snapmax"], o["floor"])   # 정수 스냅은 맨 끝

    out = tpl.copy()
    out[items] = out[items].astype(float)
    off = 0
    for t in range(C.N_TEST):
        blk = p[off:off + C.HORIZON * n].reshape(C.HORIZON, n)
        off += C.HORIZON * n
        for h in range(1, C.HORIZON + 1):
            out.loc[out.index[out[rk] == f"TEST_{t:02d}+{h}일"][0], items] = blk[h - 1]
    out[items] = out[items].round(2)

    path = os.path.join(C.SUBMISSIONS, f"submission_{stamp}.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    v = out[items].values.astype(float)
    assert not np.isnan(v).any(), "NaN 발생"
    assert (v >= o["floor"] - 1e-9).all(), "하한 미달 값 존재"

    desc = (f"3구간 {o['seg']} 경계 {o['t']}" if o["seg"] else f"배율 {o['mult']}")
    if o["snap"] != "none":
        desc += f" · 스냅 {o['snap']}"
        if np.isfinite(o["snapmax"]):
            desc += f"(p<={o['snapmax']:g})"
    print(f"저장: submission_{stamp}.csv   ({desc} · 하한 {o['floor']})")
    print(f"  min {v.min():.2f} / 중앙값 {np.median(v):.2f} / "
          f"평균 {v.mean():.2f} / max {v.max():.2f}")
    print(f"  v8(무보정) 대비 바뀐 칸 {100*(np.abs(p-cur) > 1e-9).mean():.1f}%"
          f" · 하한으로 눌린 칸 {100*(p <= o['floor']+1e-9).mean():.1f}%")
    if o["snap"] != "none":
        print(f"  스냅이 바꾼 칸 {100*(np.abs(p-pre_snap) > 1e-9).mean():.1f}%"
              f" · 정수인 칸 {100*(np.abs(p-np.round(p)) < 1e-9).mean():.1f}%")


if __name__ == "__main__":
    main()
