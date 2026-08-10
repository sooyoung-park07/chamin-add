# -*- coding: utf-8 -*-
"""Tier B — TB5: OpenAI 메뉴명 임베딩 수집.

193개 품목의 '메뉴명'(영업장 접두어 제외)만 임베딩한다 — `features.py`의 `name_cluster`가
이미 메뉴명만으로 철자 유사도를 재는 것과 같은 대상이라, 비교가 공정하다.

출력: data/tierb/menu_embeddings.npy — shape (193, dim), 행 순서는 D.item_order()와 동일
(정렬이 어긋나면 features.py의 _embed_nn()이 로드 시 행 수 불일치로 예외를 던진다).
"""
import json
import os
import urllib.request

import numpy as np

import config as C
import dataio as D

MODEL = "text-embedding-3-small"
URL = "https://api.openai.com/v1/embeddings"
OUT = os.path.join(C.DATA, "tierb", "menu_embeddings.npy")


def _load_key():
    env_path = os.path.join(C.ROOT, ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("OPENAI_API_KEY가 .env에 없음")


def embed(key, texts):
    body = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode("utf-8"))
    # 응답 순서는 요청 순서와 동일함이 보장됨(OpenAI 문서).
    return np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)


def main():
    key = _load_key()
    items = D.item_order()
    menus = [k.split("_", 1)[1] for k in items]
    print(f"품목 {len(items)}개 임베딩 요청 중 (model={MODEL})...", flush=True)
    emb = embed(key, menus)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, emb)
    print(f"저장: {OUT} · shape {emb.shape}")


if __name__ == "__main__":
    main()
