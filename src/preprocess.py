# -*- coding: utf-8 -*-
"""전처리 변형 — Phase 2 A/B 실험 대상.

각 함수는 (품목 × 날짜) 행렬을 받아 변형된 행렬을 돌려준다.
**캡 기준값은 반드시 '해당 폴드의 학습 구간'에서만 계산**해서 미래 누수를 막는다.
"""
import numpy as np


def item_caps(mat, upto_col, q=0.99, min_cap=10.0):
    """품목별 상한값 — upto_col 이전 데이터의 양수값 q분위.

    upto_col : 이 열 인덱스 미만만 사용 (폴드 학습 구간 끝)
    """
    sub = mat[:, :upto_col]
    caps = np.full(mat.shape[0], np.inf, dtype=np.float32)
    for i in range(mat.shape[0]):
        v = sub[i][sub[i] > 0]
        if len(v) >= 20:                       # 표본이 너무 적으면 캡 안 씌움
            caps[i] = max(np.quantile(v, q), min_cap)
    return caps


def apply_caps(mat, caps):
    return np.minimum(mat, caps[:, None]).astype(np.float32)


def negatives(mat, mode="keep"):
    """음수(환불) 처리.

    프로파일링 결과 음수는 전체의 0.014%(14행), 점수 기여 상한 0.0006 →
    사실상 무시 가능. 기본은 'keep'(그대로 두고 학습 타깃에서만 제외).
    """
    if mode == "keep":
        return mat
    if mode == "zero":
        return np.maximum(mat, 0).astype(np.float32)
    if mode == "abs":
        return np.abs(mat).astype(np.float32)
    raise ValueError(mode)
