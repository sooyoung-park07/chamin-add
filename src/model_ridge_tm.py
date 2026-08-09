# -*- coding: utf-8 -*-
"""팀원 Ridge 모델의 우리 검증 하네스 이식판.

원본: teammate_repo/src/final_submission.py (JongeunSong, MIT)
      실측 Public 0.491323 / Private 0.495661

**왜 이식하나** — 앙상블이 이득이 되려면 조건이 `ρ < σ_ours / σ_theirs` 다.
ρ(잔차 상관)를 재려면 두 모델이 **같은 폴드·같은 행**에서 예측해야 하는데,
원본은 전체학습 → 테스트예측 한 방향뿐이라 OOF 가 안 나온다.
그래서 피처 빌더와 학습부를 우리 폴드 루프에서 부를 수 있게 뜯어낸다.

**원본 대비 의도적 차이 — 전부 여기 적는다**
  ① origin 목록을 우리 것으로 통일. 원본은 index 28 부터("하루 여유"), 우리는 27 부터.
     행 순서를 `F.build_samples` 와 1:1 맞추기 위함이며, 이게 ρ 계산의 전제다.
  ② 범주형 12종을 문자열 대신 **정수 코드**로 넣는다. 원핫 결과는 열 순열만 다르고
     Ridge 해는 열 순열에 불변이므로 수치적으로 동일하다. 대신 문자열 500k개 ×
     4열 생성을 피해 메모리 ~500MB·시간을 아낀다.
  ③ `positive_median_columns` / `days_since_last_positive` 를 벡터화했다.
     각각 nanmedian, 역순 argmax 로 **정의상 동일**하다(루프판과 대조 테스트 있음).
  ④ 그 외 — 피처 정의 51종, alpha=10, 표본가중(power 0.5), log1p 양수학습,
     Ridge 0.68 + 같은요일양수평균 0.32, 하한 1.0 — 은 원본 그대로다.

**원본에 있으나 여기서 안 하는 것**: 전역 ×0.94. 우리 3구간 배율이 그 역할을
대신하므로, 이 모듈은 배율 이전의 **raw** 를 돌려준다. 후처리는 바깥에서 한 번만 건다.
"""
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WINDOW = 28
HORIZON = 7

RIDGE_ALPHA = 10.0
ITEM_BALANCE_POWER = 0.5
PREDICTION_FLOOR = 1.0
MAX_LOG_PRED = np.log1p(1_000_000.0)
RIDGE_WEIGHT = 0.68
BASELINE_WEIGHT = 0.32

# 원본 코드에 박혀 있는 공휴일 목록 그대로. 우리 config.HOLIDAYS 와 다를 수 있으나
# **공휴일 정의도 그들 모델의 일부**라 건드리지 않는다.
HOLIDAY_STRINGS = [
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06",
    "2023-08-15", "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02",
    "2023-10-03", "2023-10-09", "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15",
    "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18",
    "2024-10-01", "2024-10-03", "2024-10-09", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06",
]
HOLIDAYS = {pd.Timestamp(x).normalize() for x in HOLIDAY_STRINGS}

STORE_CLUSTER = {
    "화담숲주막": "화담숲", "화담숲카페": "화담숲",
    "포레스트릿": "스키겨울", "카페테리아": "스키겨울",
    "느티나무 셀프BBQ": "야외그린", "연회장": "B2B",
    "담하": "상시", "라그로타": "상시", "미라시아": "상시",
}

# 원본 NUMERIC_FEATURES 와 같은 순서 (51개)
NUM_KEYS = [
    "w_mean", "w_std", "w_max", "w_median", "w_nzratio", "w_posmean",
    "w_posmedian", "w_last", "w_last7", "w_last14", "w_prev7", "w_trend",
    "w_days_since", "w_last7_nz",
    "d_mean", "d_max", "d_posmean", "d_nzratio", "d_last",
    "st_closed_ratio", "st_closed_last7", "w_mean_open", "w_nzratio_open",
    "d_mean_open", "st_days_since_open",
    "x_store_last7", "x_store_prev7", "x_store_trend", "x_store_dow",
    "x_share_win", "x_share_dow", "st_mean", "st_nz",
    "sd_lastweek", "horizon_num",
    "is_weekend", "is_holiday", "is_holiday_eve", "is_dayoff",
    "dayoff_run_length", "dayoff_run_position", "doy_sin", "doy_cos",
    "hwadam_open", "ski_season", "ski_peak", "foliage",
    "days_to_hwadam_open", "days_to_hwadam_close", "days_to_ski_close",
    "is_high_weight",
]
# 원본 CATEGORICAL_FEATURES 와 같은 순서 (12개). 값은 정수 코드(위 ② 참조).
CAT_KEYS = [
    "item_id", "store", "cluster", "dow_cat", "month_cat", "horizon_cat",
    "item_dow", "item_horizon", "store_dow", "store_horizon",
    "cluster_dow", "cluster_month",
]
N_NUM, N_CAT = len(NUM_KEYS), len(CAT_KEYS)


# ─────────────────────────────────────────────────────────────── 보조 함수

def _positive_mean(v, axis=0):
    v = np.asarray(v, float)
    cnt = np.sum(v > 0, axis=axis)
    tot = np.sum(v, axis=axis)
    return np.divide(tot, cnt, out=np.zeros_like(tot, dtype=float), where=cnt > 0)


def _positive_median_columns(v):
    """열별 '양수만의 중앙값'. 양수가 없으면 0. (원본 루프판과 정의상 동일)"""
    v = np.where(np.asarray(v, float) > 0, v, np.nan)
    with warnings.catch_warnings():        # 전열 0인 품목 = All-NaN slice → 0 으로 간다
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(v, axis=0)
    return np.nan_to_num(out, nan=0.0)


def _nanmean_zero(v, axis=0):
    v = np.asarray(v, float)
    cnt = np.sum(np.isfinite(v), axis=axis)
    tot = np.nansum(v, axis=axis)
    return np.divide(tot, cnt, out=np.zeros_like(tot, dtype=float), where=cnt > 0)


def _days_since_last_positive(v):
    """마지막 양수로부터 며칠 지났나. 양수가 없으면 행 수. (원본과 동일)"""
    mask = np.asarray(v) > 0
    rev = mask[::-1]
    return np.where(rev.any(axis=0), rev.argmax(axis=0), mask.shape[0]).astype(float)


def _is_dayoff(d):
    d = pd.Timestamp(d).normalize()
    return d.dayofweek >= 5 or d in HOLIDAYS


def _dayoff_run(d):
    d = pd.Timestamp(d).normalize()
    if not _is_dayoff(d):
        return 0, 0
    s = e = d
    for _ in range(10):
        if _is_dayoff(s - pd.Timedelta(days=1)):
            s -= pd.Timedelta(days=1)
        else:
            break
    for _ in range(10):
        if _is_dayoff(e + pd.Timedelta(days=1)):
            e += pd.Timedelta(days=1)
        else:
            break
    return (e - s).days + 1, (d - s).days + 1


def _signed_days_to(d, month, day):
    d = pd.Timestamp(d).normalize()
    cands = [pd.Timestamp(year=y, month=month, day=day)
             for y in (d.year - 1, d.year, d.year + 1)]
    return float(min(((c - d).days for c in cands), key=abs))


# ─────────────────────────────────────────────────────────────── 컨텍스트

class TMContext:
    """품목 순서·영업장/군집 코드. items 는 sample_submission 열 순서(우리와 동일)."""

    def __init__(self, items):
        self.items = list(items)
        self.n = len(self.items)
        stores = [str(k).split("_", 1)[0] for k in self.items]
        self.store_names = sorted(set(stores))
        s2c = {s: i for i, s in enumerate(self.store_names)}
        self.store_code = np.array([s2c[s] for s in stores], dtype=np.int32)
        self.n_store = len(self.store_names)

        clusters = [STORE_CLUSTER.get(s, "기타") for s in stores]
        self.cluster_names = sorted(set(clusters))
        c2c = {c: i for i, c in enumerate(self.cluster_names)}
        self.cluster_code = np.array([c2c[c] for c in clusters], dtype=np.int32)
        self.n_cluster = len(self.cluster_names)

        self.is_high_weight = np.array(
            [1.0 if s in ("담하", "미라시아") else 0.0 for s in stores])
        self.item_code = np.arange(self.n, dtype=np.int32)


# ─────────────────────────────────────────────────────── 창 하나 → 피처 행렬

def _window_rows(ctx, hist_dates, x, target_date, horizon):
    """(28, n_items) 창 하나 + 타깃일 → (num[n,51], cat[n,12]).

    원본 build_window_rows 와 같은 계산. x 는 음수 클립(>=0) 된 값이어야 한다.
    """
    n = ctx.n
    sc = ctx.store_code
    target_date = pd.Timestamp(target_date)

    # 영업장 일별 합계 → 품목마다 자기 영업장 값
    store_x = np.zeros((WINDOW, ctx.n_store))
    np.add.at(store_x.T, sc, x.T)
    item_store_x = store_x[:, sc]
    item_store_open = item_store_x > 0

    last7, last14, prev7 = x[-7:], x[-14:], x[-14:-7]
    w_last7, w_prev7 = last7.mean(0), prev7.mean(0)

    same_dow = np.asarray(hist_dates.dayofweek == target_date.dayofweek)
    d = x[same_dow]

    open_x = np.where(item_store_open, x, np.nan)
    d_open = np.where(item_store_open[same_dow], d, np.nan)
    store_open = store_x > 0
    store_days_since = _days_since_last_positive(store_x)

    store_last7 = store_x[-7:].mean(0)
    store_prev7 = store_x[-14:-7].mean(0)
    store_dow = store_x[same_dow].mean(0)
    share = np.divide(x, item_store_x, out=np.zeros_like(x), where=item_store_x > 0)

    lag7_pos = int(np.flatnonzero(hist_dates == target_date - pd.Timedelta(days=7))[0])

    doy = target_date.dayofyear
    ylen = 366 if target_date.is_leap_year else 365
    run_len, run_pos = _dayoff_run(target_date)
    hwadam = int((target_date.month > 3 or (target_date.month == 3 and target_date.day >= 20))
                 and (target_date.month < 11 or (target_date.month == 11 and target_date.day <= 30)))
    ski = int(target_date.month in (12, 1, 2)
              or (target_date.month == 3 and target_date.day <= 5))
    ski_peak = int((target_date.month == 12 and target_date.day >= 20) or target_date.month == 1)
    foliage = int((target_date.month == 10 and target_date.day >= 15)
                  or (target_date.month == 11 and target_date.day <= 20))
    rep = lambda v: np.full(n, float(v))

    cols = {
        "w_mean": x.mean(0), "w_std": x.std(0), "w_max": x.max(0),
        "w_median": np.median(x, 0), "w_nzratio": (x > 0).mean(0),
        "w_posmean": _positive_mean(x), "w_posmedian": _positive_median_columns(x),
        "w_last": x[-1], "w_last7": w_last7, "w_last14": last14.mean(0),
        "w_prev7": w_prev7, "w_trend": np.divide(w_last7 + 0.25, w_prev7 + 0.25),
        "w_days_since": _days_since_last_positive(x),
        "w_last7_nz": (last7 > 0).mean(0),
        "d_mean": d.mean(0), "d_max": d.max(0), "d_posmean": _positive_mean(d),
        "d_nzratio": (d > 0).mean(0), "d_last": d[-1],
        "st_closed_ratio": 1.0 - store_open.mean(0)[sc],
        "st_closed_last7": 1.0 - store_open[-7:].mean(0)[sc],
        "w_mean_open": _nanmean_zero(open_x),
        "w_nzratio_open": np.divide(
            np.sum((x > 0) & item_store_open, 0), np.sum(item_store_open, 0),
            out=np.zeros(n), where=np.sum(item_store_open, 0) > 0),
        "d_mean_open": _nanmean_zero(d_open),
        "st_days_since_open": store_days_since[sc],
        "x_store_last7": store_last7[sc], "x_store_prev7": store_prev7[sc],
        "x_store_trend": ((store_last7 + 1.0) / (store_prev7 + 1.0))[sc],
        "x_store_dow": store_dow[sc],
        "x_share_win": _nanmean_zero(np.where(item_store_open, share, np.nan)),
        "x_share_dow": _nanmean_zero(np.where(item_store_open[same_dow],
                                              share[same_dow], np.nan)),
        "st_mean": store_x.mean(0)[sc], "st_nz": store_open.mean(0)[sc],
        "sd_lastweek": x[lag7_pos], "horizon_num": rep(horizon),
        "is_weekend": rep(target_date.dayofweek >= 5),
        "is_holiday": rep(target_date.normalize() in HOLIDAYS),
        "is_holiday_eve": rep((target_date + pd.Timedelta(days=1)).normalize() in HOLIDAYS),
        "is_dayoff": rep(_is_dayoff(target_date)),
        "dayoff_run_length": rep(run_len), "dayoff_run_position": rep(run_pos),
        "doy_sin": rep(np.sin(2 * np.pi * doy / ylen)),
        "doy_cos": rep(np.cos(2 * np.pi * doy / ylen)),
        "hwadam_open": rep(hwadam), "ski_season": rep(ski),
        "ski_peak": rep(ski_peak), "foliage": rep(foliage),
        "days_to_hwadam_open": rep(_signed_days_to(target_date, 3, 20)),
        "days_to_hwadam_close": rep(_signed_days_to(target_date, 11, 30)),
        "days_to_ski_close": rep(_signed_days_to(target_date, 3, 5)),
        "is_high_weight": ctx.is_high_weight,
    }
    num = np.column_stack([cols[k] for k in NUM_KEYS]).astype(np.float32)

    dow, mon, h = target_date.dayofweek, target_date.month, int(horizon)
    ic, stc, clc = ctx.item_code, ctx.store_code, ctx.cluster_code
    cat = np.column_stack([
        ic, stc, clc, np.full(n, dow), np.full(n, mon), np.full(n, h),
        ic * 7 + dow, ic * 7 + (h - 1),
        stc * 7 + dow, stc * 7 + (h - 1),
        clc * 7 + dow, clc * 12 + (mon - 1),
    ]).astype(np.int32)
    return num, cat


def build_from_window(ctx, hist_dates, hist_mat):
    """28일 창 하나 (품목 × 28) → (num, cat) for h=1..7. 테스트 창 전용."""
    d = pd.DatetimeIndex(hist_dates)
    x = np.maximum(np.asarray(hist_mat, float), 0.0).T      # (28, n_items)
    nums, cats = [], []
    for h in range(1, HORIZON + 1):
        nu, ca = _window_rows(ctx, d, x, d[-1] + pd.Timedelta(days=h), h)
        nums.append(nu)
        cats.append(ca)
    return np.concatenate(nums), np.concatenate(cats)


def build_frames(ctx, mat, dates, origins):
    """우리 origin 목록 → (num, cat, y, item_idx). 행 순서 = F.build_samples 와 동일.

    mat 은 (품목 × 날짜) 원본. 피처에는 음수 클립본을 쓰고(원본 규약),
    y 는 클립 전 원값을 돌려준다(채점은 원값으로 한다).
    """
    dates = pd.DatetimeIndex(dates)
    xall = np.maximum(np.asarray(mat, float), 0.0)      # 피처용: 음수 → 0
    nums, cats, ys = [], [], []
    for o in origins:
        hd = dates[o - WINDOW + 1: o + 1]
        x = xall[:, o - WINDOW + 1: o + 1].T             # (28, n_items)
        for h in range(1, HORIZON + 1):
            nu, ca = _window_rows(ctx, hd, x, dates[o] + pd.Timedelta(days=h), h)
            nums.append(nu)
            cats.append(ca)
            ys.append(np.asarray(mat[:, o + h], float))
    item_idx = np.tile(np.arange(ctx.n), len(origins) * HORIZON)
    return (np.concatenate(nums), np.concatenate(cats),
            np.concatenate(ys), item_idx)


# ─────────────────────────────────────────────────────────────── 학습·예측

def _make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=False))])
    categorical = OneHotEncoder(handle_unknown="ignore", dtype=np.float32,
                                sparse_output=True)
    return ColumnTransformer(
        [("num", numeric, list(range(N_NUM))),
         ("cat", categorical, list(range(N_NUM, N_NUM + N_CAT)))],
        sparse_threshold=1.0)


def _training_weights(item_idx, store_code, power=ITEM_BALANCE_POWER):
    """원본 training_weights 와 동일 — 품목별 표본수 보정 × 영업장 품목수 보정."""
    n_pos = np.bincount(item_idx, minlength=len(store_code))[item_idx].astype(float)
    item_comp = np.power(np.median(n_pos) / np.maximum(n_pos, 1.0), power)

    st = store_code[item_idx]
    items_per_store = np.array(
        [len(np.unique(item_idx[st == s])) for s in range(store_code.max() + 1)],
        dtype=float)
    cnt = items_per_store[st]
    # ⚠️ 원본은 groupby.transform 결과의 median 이라 **행 가중 중앙값**이다
    #   (영업장 9개의 중앙값이 아니다). 품목 많은 영업장이 중앙값을 끌어올린다.
    store_comp = np.median(cnt) / np.maximum(cnt, 1.0)

    w = np.clip(item_comp * store_comp, 0.2, 5.0)
    return w / w.mean()


def _inverse_log(v):
    return np.maximum(np.expm1(np.clip(np.asarray(v, float), 0.0, MAX_LOG_PRED)),
                      PREDICTION_FLOOR)


def fit_predict(ctx, tr_num, tr_cat, tr_y, tr_item, va_num, va_cat,
                alpha=RIDGE_ALPHA):
    """양수 행으로 Ridge 학습 → 검증 raw 예측 3종 반환.

    returns dict(ridge=, baseline=, blend=)  — 전부 하한 1.0, 전역 ×0.94 **미적용**.
    """
    m = tr_y > 0
    Xn, Xc = tr_num[m], tr_cat[m]
    y = np.log1p(tr_y[m])
    w = _training_weights(tr_item[m], ctx.store_code)

    pre = _make_preprocessor()
    Xtr = pre.fit_transform(np.hstack([Xn, Xc.astype(np.float32)]))
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=3000)
    model.fit(Xtr, y, sample_weight=w)

    Xva = pre.transform(np.hstack([va_num, va_cat.astype(np.float32)]))
    ridge = _inverse_log(model.predict(Xva))
    baseline = np.maximum(va_num[:, NUM_KEYS.index("d_posmean")].astype(float), 1.0)
    blend = np.maximum(RIDGE_WEIGHT * ridge + BASELINE_WEIGHT * baseline,
                       PREDICTION_FLOOR)
    return {"ridge": ridge, "baseline": baseline, "blend": blend,
            "design_shape": Xtr.shape}
