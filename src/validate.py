# -*- coding: utf-8 -*-
"""검증 체계 — 대회 구조(28일창 → +1~+7일)를 그대로 재현한 롤링-오리진.

폴드는 계절별 3개. 각 폴드의 검증 origin은 7일 간격(예측구간이 안 겹치게).
학습은 그 폴드 시작 이전 데이터만 사용(미래 누수 차단).
"""
import numpy as np
import pandas as pd

import config as C
import dataio as D
from metrics import competition_score, flat_smape, make_weights

# 계절별 3폴드 (창 마지막날 기준 구간)
FOLDS = [
    ("F1 가을", "2023-08-25", "2023-11-23"),
    ("F2 겨울", "2023-11-24", "2024-02-22"),
    ("F3 봄",   "2024-02-23", "2024-06-08"),
]


def origins(dates, d0, d1, n_days, step=7):
    """창 마지막날이 [d0,d1]이고 +7일 타깃이 존재하는 origin 인덱스.

    step=7 (기본) : 예측구간이 안 겹치게 띄운다. 폴드당 13~16개.
    step=1        : 매일. 폴드당 91~107개 (채점행 7배).

    ※ step=1로 촘촘하게 재보는 실험을 했으나 **노이즈가 전혀 줄지 않았다**
      (σ 0.0014 → 0.0015). 실행시간만 6배(85s → 470s).
      이유: 노이즈의 출처가 '표본'이 아니라 '모델'이다. 시드가 바뀌면 특정 품목을
      모든 날짜에서 체계적으로 다르게 예측하므로, 상관 높은 날짜를 더 넣어도
      상쇄되지 않는다. **노이즈를 줄이려면 시드 수를 늘려야 한다.**
      → 스크리닝은 step=7로 빠르게, 최종 확인만 step=1로.
    """
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    idx = [o for o in range(C.WINDOW - 1, n_days - C.HORIZON)
           if d0 <= dates[o] <= d1]
    return idx[::step]


def train_origins(dates, d0, n_days):
    """폴드 시작 이전의 학습용 origin (타깃이 검증구간에 안 닿도록)."""
    d0 = pd.Timestamp(d0)
    return [o for o in range(C.WINDOW - 1, n_days - C.HORIZON) if dates[o] < d0]


def collect_targets(mat, origin_list):
    """검증용 정답 y와 품목 인덱스를 origin 순서대로 평탄화."""
    n_item = mat.shape[0]
    ys, iids, hs = [], [], []
    for o in origin_list:
        for h in range(1, C.HORIZON + 1):
            ys.append(mat[:, o + h])
            iids.append(np.arange(n_item))
            hs.append(np.full(n_item, h))
    return (np.concatenate(ys), np.concatenate(iids), np.concatenate(hs))


class Evaluator:
    """예측 배열을 받아 공식 지표 / 비교 지표 / 분해를 한 번에 계산."""

    def __init__(self, items):
        self.items = items
        self.item_store = np.array([k.split("_")[0] for k in items])
        self.n_items = len(items)

    def score(self, y, p, iids, high=2.0):
        """단일 점수 (실험 루프에서 쓰는 기본 지표)."""
        return competition_score(y, p, iids, self.item_store,
                                 make_weights(high), self.n_items)

    def full(self, y, p, iids):
        """공식지표(가중 민감도 3종) + flat 비교 + 영업장별 분해."""
        out = {}
        for h in (1.0,) + C.WEIGHT_CANDIDATES:
            lbl = "균등" if h == 1.0 else f"담하·미라시아 {h:g}x"
            out[lbl] = competition_score(y, p, iids, self.item_store,
                                         make_weights(h), self.n_items)
        out["flat(참고)"] = flat_smape(y, p)
        _, per_store, item_score, item_cnt = competition_score(
            y, p, iids, self.item_store, None, self.n_items, return_parts=True)
        return out, per_store, item_score, item_cnt


def rule_baseline(mat, dates, origin_list):
    """베이스라인: 창 안 '같은 요일의 양수 평균' (없으면 창 전체 양수평균)."""
    preds = []
    for o in origin_list:
        win = mat[:, o - C.WINDOW + 1:o + 1]
        wd = np.array([d.dayofweek for d in dates[o - C.WINDOW + 1:o + 1]])
        pos = win > 0
        cnt = pos.sum(1)
        fallback = np.where(cnt > 0, (win * pos).sum(1) / np.maximum(cnt, 1), 0.0)
        for h in range(1, C.HORIZON + 1):
            td = dates[o] + pd.Timedelta(days=h)
            sel = win[:, wd == td.dayofweek]
            sp = sel > 0
            sc = sp.sum(1)
            preds.append(np.where(sc > 0, (sel * sp).sum(1) / np.maximum(sc, 1),
                                  fallback))
    return np.concatenate(preds)


def main():
    tr = D.load_train()
    items = D.item_order()
    mat, dates = D.to_matrix(tr, items)
    ev = Evaluator(items)
    nd = mat.shape[1]

    print("=" * 76)
    print("Phase 1 검증기 — 공식 지표 vs 기존에 쓰던 flat 지표")
    print("=" * 76)

    agg = {}
    for name, d0, d1 in FOLDS:
        va = origins(dates, d0, d1, nd)
        y, iids, _ = collect_targets(mat, va)
        p = np.maximum(rule_baseline(mat, dates, va), 1.0)   # 하한 1 적용
        res, per_store, item_score, item_cnt = ev.full(y, p, iids)
        print(f"\n[{name}]  origin {len(va)}개 · 유효행 {int((y != 0).sum()):,}")
        for k, v in res.items():
            print(f"   {k:<20s} {v:.4f}")
            agg.setdefault(k, []).append(v)

    print("\n" + "-" * 76)
    print("폴드 평균")
    for k, v in agg.items():
        print(f"   {k:<20s} {np.mean(v):.4f}   ({' '.join(f'{x:.4f}' for x in v)})")

    # 마지막 폴드 기준 분해
    print("\n[영업장별 (마지막 폴드)]")
    for s, v in sorted(per_store.items(), key=lambda x: -x[1]):
        mark = " ★가중높음" if s in C.HIGH_WEIGHT_STORES else ""
        n = (ev.item_store == s).sum()
        print(f"   {s:<14s} {v:.4f}   품목 {n}개{mark}")

    print("\n[품목 단위 취급 확인 — 유효날짜 수와 무관하게 1표]")
    order = np.argsort(-np.nan_to_num(item_score, nan=-1))
    print("   오차 최악 5개:")
    for i in order[:5]:
        print(f"     {items[i]:<38s} SMAPE {item_score[i]:.3f}  유효날짜 {item_cnt[i]:>4d}")
    print("   오차 최선 5개:")
    for i in order[::-1][:5]:
        if np.isnan(item_score[i]):
            continue
        print(f"     {items[i]:<38s} SMAPE {item_score[i]:.3f}  유효날짜 {item_cnt[i]:>4d}")


if __name__ == "__main__":
    main()
