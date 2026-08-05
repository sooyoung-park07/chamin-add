# -*- coding: utf-8 -*-
"""평가 지표 — 대회 공식 수식 그대로 구현.

    Score = Σ_s  w_s · ( 1/|I_s| · Σ_{i∈I_s} ( 1/T_i · Σ_t 2|A-P| / (|A|+|P|) ) )

    s   : 식음업장,  I_s: 그 업장의 품목 집합
    T_i : 품목 i에서 **A ≠ 0** 인 날짜 수  ← 0보다 큰 게 아니라 '0이 아닌'
    w_s : 업장 가중치 (비공개. 담하·미라시아가 높음)

구조상 세 번 평균한다:
  ① 품목 안: 유효 날짜끼리 평균  → 날짜 많은 품목이 유리하지 않음
  ② 업장 안: 품목끼리 단순 평균  → **희귀 품목도 주력 품목과 같은 1표**
  ③ 업장끼리: w_s 가중합

주의: 단순히 전체 행을 평균내는 방식(flat)은 주력 품목에 점수가 쏠려
      실제보다 후하게 나온다. 비교용으로 flat_smape도 남겨둔다.
"""
import numpy as np

import config as C


def _terms(a, p):
    """행별 SMAPE 항. a==0인 행은 제외(마스크와 함께 반환)."""
    a = np.asarray(a, dtype=np.float64)
    p = np.maximum(np.asarray(p, dtype=np.float64), 0.0)
    valid = a != 0                       # 공식 정의: A != 0 (음수도 포함)
    denom = np.abs(a) + np.abs(p)
    t = np.zeros_like(a)
    nz = valid & (denom > 0)
    t[nz] = 2.0 * np.abs(a[nz] - p[nz]) / denom[nz]
    # A!=0 인데 denom==0 은 불가능(|A|>0). 안전장치로 남김.
    return t, valid


def flat_smape(a, p):
    """비교용: 유효 행 전체를 평평하게 평균 (공식 지표 아님)."""
    t, valid = _terms(a, p)
    return float(t[valid].mean()) if valid.any() else np.nan


def per_item_smape(a, p, item_ids, n_items=None):
    """품목별 평균 SMAPE. 유효 날짜가 하나도 없는 품목은 NaN."""
    a, p, item_ids = np.asarray(a), np.asarray(p), np.asarray(item_ids)
    n_items = n_items or int(item_ids.max()) + 1
    t, valid = _terms(a, p)
    s = np.bincount(item_ids[valid], weights=t[valid], minlength=n_items)
    c = np.bincount(item_ids[valid], minlength=n_items)
    out = np.full(n_items, np.nan)
    np.divide(s, c, out=out, where=c > 0)
    return out, c


def competition_score(a, p, item_ids, item_store, store_weights=None,
                      n_items=None, return_parts=False):
    """대회 공식 지표.

    item_ids   : 각 행이 어느 품목인지 (정수 인덱스)
    item_store : 품목 인덱스 -> 영업장 이름 (길이 n_items 리스트/배열)
    store_weights : {영업장명: 가중치}. None이면 균등. 내부에서 합=1로 정규화.
    """
    item_store = np.asarray(item_store)
    n_items = n_items or len(item_store)
    item_score, item_cnt = per_item_smape(a, p, item_ids, n_items)

    stores = sorted(set(item_store.tolist()))
    if store_weights is None:
        store_weights = {s: 1.0 for s in stores}
    w = np.array([store_weights.get(s, 1.0) for s in stores], dtype=np.float64)

    per_store, used = {}, np.zeros(len(stores), dtype=bool)
    for k, s in enumerate(stores):
        m = (item_store == s) & ~np.isnan(item_score)   # T_i=0 품목은 제외
        if m.sum() == 0:
            continue
        per_store[s] = float(item_score[m].mean())      # 1/|I_s|
        used[k] = True

    ww = w[used] / w[used].sum()                        # 합=1로 정규화
    vals = np.array([per_store[s] for k, s in enumerate(stores) if used[k]])
    score = float((ww * vals).sum())
    if return_parts:
        return score, per_store, item_score, item_cnt
    return score


def make_weights(high=2.0):
    """담하·미라시아에 high배 가중을 준 가중치 dict."""
    return {s: (high if s in C.HIGH_WEIGHT_STORES else 1.0)
            for s in C.STORE_CLUSTER}
