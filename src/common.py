"""Shared data loading for the MA289 word-embedding competition."""
import os
import numpy as np
import pandas as pd

DATA = os.environ.get("MA289_DATA")

def load():
    ctx = pd.read_csv(os.path.join(DATA, "context_train.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    tokens = sorted(set(ctx["center_token"]) | set(ctx["context_token"]))
    tok2idx = {t: i for i, t in enumerate(tokens)}
    center = ctx["center_token"].map(tok2idx).to_numpy()
    context = ctx["context_token"].map(tok2idx).to_numpy()
    t1 = test["token1"].map(tok2idx).to_numpy()
    t2 = test["token2"].map(tok2idx).to_numpy()
    corpus = []
    with open(os.path.join(DATA, "train_corpus.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append([tok2idx[t] for t in line.split()])
    return dict(tokens=tokens, tok2idx=tok2idx, V=len(tokens),
                center=center, context=context,
                test=test, t1=t1, t2=t2, corpus=corpus)

def pair_cos(E, t1, t2):
    """Cosine similarity between embedding rows for each test pair."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return np.sum(En[t1] * En[t2], axis=1)
