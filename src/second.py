"""Wide-window PPMI-SVD plus second-order (similarity-of-similarity) features."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos
from ppmi2 import cooc, ppmi

def rownorm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

if __name__ == "__main__":
    D = load(); V = D["V"]; outdir = sys.argv[1]
    t1, t2 = D["t1"], D["t2"]
    for w in (5, 10, 20, 50):
        M = cooc(D["corpus"], V, w, True)
        X = ppmi(M, 0.75)
        # first-order (full PPMI rows)
        np.savez_compressed(os.path.join(outdir, f"o1_full_w{w}.npz"),
                            cos=pair_cos(X, t1, t2))
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        for d in (100, 200):
            E = U[:, :d] * s[:d]
            np.savez_compressed(os.path.join(outdir, f"o1_svd_w{w}_d{d}.npz"),
                                cos=pair_cos(E, t1, t2))
            # second order: cosine between rows of the similarity matrix
            Sm = rownorm(E) @ rownorm(E).T
            np.savez_compressed(os.path.join(outdir, f"o2_svd_w{w}_d{d}.npz"),
                                cos=pair_cos(Sm, t1, t2))
        Sfull = rownorm(X) @ rownorm(X).T
        np.savez_compressed(os.path.join(outdir, f"o2_full_w{w}.npz"),
                            cos=pair_cos(Sfull, t1, t2))
        # third order
        np.savez_compressed(os.path.join(outdir, f"o3_full_w{w}.npz"),
                            cos=pair_cos(rownorm(Sfull) @ rownorm(Sfull).T, t1, t2))
        print("w", w, "done", flush=True)
