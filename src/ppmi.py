"""PPMI + truncated SVD embeddings from the window-2 co-occurrence matrix."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos

def cooc_matrix(corpus, V, w=2, weighted=False):
    M = np.zeros((V, V), dtype=np.float64)
    for line in corpus:
        a = np.asarray(line)
        n = len(a)
        for off in range(1, w + 1):
            wt = 1.0 / off if weighted else 1.0
            np.add.at(M, (a[:n - off], a[off:]), wt)
            np.add.at(M, (a[off:], a[:n - off]), wt)
    return M

def ppmi(M, alpha=0.75, shift=0.0):
    total = M.sum()
    row = M.sum(1, keepdims=True)
    col = M.sum(0, keepdims=True) ** alpha
    col = col / col.sum() * 1.0
    P = M / total
    Pr = row / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(P / (Pr * col))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi - shift, 0.0)

def svd_embed(X, d, p=0.5):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :d] * (s[:d] ** p)

if __name__ == "__main__":
    D = load(); V = D["V"]
    outdir = sys.argv[1]
    for weighted in (False, True):
        M = cooc_matrix(D["corpus"], V, 2, weighted)
        for alpha in (1.0, 0.75):
            X = ppmi(M, alpha)
            for d in (24, 50, 100, 300):
                E = svd_embed(X, d)
                c = pair_cos(E, D["t1"], D["t2"])
                tag = f"ppmi_w{int(weighted)}_a{alpha}_d{d}"
                np.savez_compressed(os.path.join(outdir, tag + ".npz"), cos=c, E=E)
                print(f"{tag}: mean={c.mean():.3f} sd={c.std():.3f}", flush=True)
