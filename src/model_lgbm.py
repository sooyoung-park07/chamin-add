# -*- coding: utf-8 -*-
"""LightGBM 모델 — 학습/예측 규약을 한곳에.

확정된 설계 (예비실험 근거):
  - **y != 0 인 행만 학습** : 실측 0은 채점 제외 → 0을 섞으면 예측이 내려앉음(0.895→0.543)
  - log1p 타깃 + L1 목적함수 : 우편향 수요에 적합, SMAPE와 정렬
  - 예측 하한 1.0 : 채점행은 |A|>=1 이라 1 미만 예측은 지배당함(증명·검증 완료)
  - 시드 앙상블
"""
import numpy as np
import lightgbm as lgb

import features as F

# ⚠️⚠️ 되돌림 (2026-08-02). Phase 5-c 재튜닝값을 잠시 넣었다가 **LB에서 확인 후 원복**했다.
#
#   재튜닝값 leaves511 · mdl10 · ff0.65 · bagging끄기 · L2:3 · L1:2
#     → CV(F2+F3) 0.4709 → 0.4640 으로 **CV 1위**
#     → 그런데 실제 LB 0.4818 → **0.487 로 꼴찌** (v4 제출)
#
# 즉 하이퍼파라미터 탐색이 **CV에 과적합**했다. 확인 시드 분리로도 못 걸렀는데,
# 시드 분리는 '시드 노이즈'만 막고 'CV와 LB의 분포 차이'는 못 막기 때문이다.
# 상세: experiments/log.md 의 Phase 5-e.
#
# ★ 규칙: **모형 용량을 키우는 방향의 CV 개선은 믿지 말 것.**
#    관측 4건에서 CV↔LB 격차가 용량과 함께 단조 증가했다(+0.009 → +0.023).
PARAMS = dict(
    objective="regression_l1", metric="l1", learning_rate=0.05,
    num_leaves=255, min_data_in_leaf=40, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
    verbosity=-1,
)
ROUNDS = 1000
SEEDS = (42, 7, 2024, 913, 31)
FLOOR = 1.0


def fit(X, y, feature_names, seeds=SEEDS, params=None, rounds=ROUNDS,
        cat_features=None, n_threads=None):
    """y != 0 인 행만 사용해 시드별 모델 리스트를 반환."""
    import os
    p = dict(PARAMS, **(params or {}))
    p["num_threads"] = n_threads or os.cpu_count()
    m = y != 0
    Xt, yt = X[m], np.maximum(y[m], 1.0)      # 음수(환불)는 1로 (예측 불가 대상)
    # ⚠️ 범주형 목록은 **실제로 넘긴 열 안에 있는 것만** 남긴다.
    #   F.CATEGORICAL 에는 기각된 `name_cluster` 가 들어 있는데 활성 57개에는 없다.
    #   거르지 않으면 lgb.Dataset 이 "없는 피처를 범주형으로 지정했다"며 거부한다.
    #   (2026-08-04 발견. 이 상태로는 fit() 이 아예 돌지 않았다.)
    cats = [c for c in (cat_features or F.CATEGORICAL) if c in feature_names]
    models = []
    for sd in seeds:
        # ⚠️ Dataset 은 설정마다 새로 만든다 — 재사용하면 min_data_in_leaf 변경이
        #   조용히 무시된다(feature_pre_filter. Phase 5-c 에서 실제로 당했다).
        ds = lgb.Dataset(Xt, label=np.log1p(yt), feature_name=feature_names,
                         categorical_feature=cats, free_raw_data=False)
        models.append(lgb.train(dict(p, seed=sd), ds, num_boost_round=rounds))
    return models


def predict(models, X, floor=FLOOR):
    p = np.mean([np.expm1(m.predict(X)) for m in models], axis=0)
    return np.maximum(p, floor)
