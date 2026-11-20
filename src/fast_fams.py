"""Fast extra families: NMF, log-count LSA, Hellinger PCA (no GloVe)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos
from ppmi2 import cooc, ppmi

def nmf(X, d, seed, iters=150):
    rng = np.random.default_rng(seed)
    W = rng.random((X.shape[0], d)) + 0.1
    H = rng.random((d, X.shape[1])) + 0.1
    for _ in range(iters):
        H *= (W.T @ X) / np.maximum(W.T @ W @ H, 1e-9)
        W *= (X @ H.T) / np.maximum(W @ H @ H.T, 1e-9)
    return W

if __name__ == "__main__":
    D = load(); V = D["V"]; outdir = sys.argv[1]
    t1, t2 = D["t1"], D["t2"]
    for w in (5, 10):
        M = cooc(D["corpus"], V, w, True)
        X = ppmi(M, 0.75)
        for d in (50, 100):
            np.savez_compressed(os.path.join(outdir, f"nmf_w{w}_d{d}.npz"),
                                cos=pair_cos(nmf(X, d, 0), t1, t2))
        U, s, _ = np.linalg.svd(np.log1p(M), full_matrices=False)
        for d in (50, 100, 200):
            np.savez_compressed(os.path.join(outdir, f"lsa_w{w}_d{d}.npz"),
                                cos=pair_cos(U[:, :d] * s[:d], t1, t2))
        Hh = np.sqrt(M / np.maximum(M.sum(1, keepdims=True), 1e-9))
        U, s, _ = np.linalg.svd(Hh - Hh.mean(0), full_matrices=False)
        for d in (50, 100):
            np.savez_compressed(os.path.join(outdir, f"hell_w{w}_d{d}.npz"),
                                cos=pair_cos(U[:, :d] * s[:d], t1, t2))
        print("w", w, "done", flush=True)
