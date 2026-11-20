"""Wide sweep of PPMI-SVD embeddings over windows, dims, weighting schemes."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos

def cooc(corpus, V, w, harmonic=True):
    M = np.zeros((V, V))
    for line in corpus:
        a = np.asarray(line); n = len(a)
        for off in range(1, w + 1):
            wt = 1.0 / off if harmonic else 1.0
            np.add.at(M, (a[:n-off], a[off:]), wt)
            np.add.at(M, (a[off:], a[:n-off]), wt)
    return M

def ppmi(M, alpha=0.75, k=1.0):
    tot = M.sum()
    pw = M.sum(1, keepdims=True) / tot
    pc = M.sum(0, keepdims=True) ** alpha
    pc = pc / pc.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((M / tot) / (pw * pc) / k)
    x[~np.isfinite(x)] = 0.0
    return np.maximum(x, 0.0)

if __name__ == "__main__":
    D = load(); V = D["V"]; outdir = sys.argv[1]
    for w in (2, 5, 10):
        for harm in (True, False):
            M = cooc(D["corpus"], V, w, harm)
            for alpha in (0.75, 1.0):
                X = ppmi(M, alpha)
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                for d in (50, 100, 200, 300):
                    for p in (0.0, 0.5, 1.0):
                        E = U[:, :d] * (s[:d] ** p)
                        c = pair_cos(E, D["t1"], D["t2"])
                        tag = f"svd_w{w}_h{int(harm)}_a{alpha}_d{d}_p{p}"
                        np.savez_compressed(os.path.join(outdir, tag + ".npz"), cos=c)
                # full PPMI rows (no SVD) as a baseline
                c = pair_cos(X, D["t1"], D["t2"])
                np.savez_compressed(os.path.join(outdir, f"ppmifull_w{w}_h{int(harm)}_a{alpha}.npz"), cos=c)
            print("done w", w, "harm", harm, flush=True)
