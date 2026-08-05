# -*- coding: utf-8 -*-
"""데이터 로드 — 모든 모듈이 이 함수들만 통해 데이터를 읽는다.

포맷 규약:
  long  : (date, key, qty) 3컬럼 DataFrame. key = "영업장명_메뉴명"
  matrix: (품목 193 × 날짜) numpy 배열 + ITEMS/DATES 인덱스. 품목 순서는
          sample_submission의 열 순서로 고정 (제출까지 순서가 절대 안 바뀜)
"""
import json
import os
import re

import numpy as np
import pandas as pd

import config as C


def _norm(s: str) -> str:
    """메뉴명 매칭용 공백 정규화 (중복 공백·앞뒤 공백 제거)."""
    return re.sub(r"\s+", " ", str(s)).strip()


def load_long(path: str) -> pd.DataFrame:
    """CSV 하나를 long 포맷으로. 컬럼명을 영문으로 통일."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = ["date", "key", "qty"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_train() -> pd.DataFrame:
    df = load_long(C.TRAIN_CSV)
    df["store"] = df["key"].str.split("_").str[0]
    df["menu"] = df["key"].str.split("_", n=1).str[1]
    return df


def load_test(i: int) -> pd.DataFrame:
    return load_long(os.path.join(C.TEST_DIR, f"TEST_{i:02d}.csv"))


def load_submission_template() -> pd.DataFrame:
    return pd.read_csv(C.SUBMISSION_TPL, encoding="utf-8-sig")


def item_order() -> list:
    """품목 정렬 기준 = 제출 파일의 열 순서 (193개)."""
    return list(load_submission_template().columns[1:])


def to_matrix(df_long: pd.DataFrame, items: list):
    """long → (품목 × 날짜) 행렬. 반환 (mat, dates).

    - 관측 없는 셀은 0 (train은 전 조합이 존재하므로 실질 영향 없음)
    - items 순서 고정
    """
    piv = df_long.pivot(index="key", columns="date", values="qty").reindex(items)
    mat = np.nan_to_num(piv.values.astype(np.float32), nan=0.0)
    return mat, list(piv.columns)


def load_labels() -> dict:
    """메뉴 라벨(가격·유형·계절힌트). 키 = 정규화된 'store_menu'."""
    with open(C.LABELS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)["items"]
    return {_norm(it["store"]) + "_" + _norm(it["menu"]): it for it in raw}


def label_of(labels: dict, key: str, field: str, default=None):
    store, menu = key.split("_", 1)
    it = labels.get(_norm(store) + "_" + _norm(menu))
    return it.get(field, default) if it else default
