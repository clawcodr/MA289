"""Build the ensemble consensus ranking and write a rank-uniform submission."""
import os, sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def normal_scores(x):
    r = pd.Series(x).rank(method="average").to_numpy()
    return norm.ppf(r / (len(x) + 1.0))

out2, outdir = sys.argv[1], sys.argv[2]
byep = {}
for f in glob.glob(os.path.join(out2, "*.npz")):
    b = os.path.basename(f)[:-4]
    ep = int(b.split("_e")[-1])
    byep.setdefault(ep, []).append(np.load(f)["cos"])

ep_cons = {}
for ep, lst in sorted(byep.items()):
    Z = np.array([normal_scores(c) for c in lst])
    ep_cons[ep] = Z.mean(0)
    print(f"epoch {ep}: {len(lst)} runs")

eps = sorted(ep_cons)
print("\ncorrelation between epoch-consensuses:")
print("        " + " ".join(f"{e:>6d}" for e in eps))
for e1 in eps:
    print(f"  e{e1:<5d} " + " ".join(
        f"{spearmanr(ep_cons[e1], ep_cons[e2]).correlation:6.3f}" for e2 in eps))

# hedge across the unknown reference epoch
W = {20: 1.0, 35: 1.2, 60: 1.2, 100: 1.0}
score = sum(W[e] * normal_scores(ep_cons[e]) for e in eps) / sum(W[e] for e in eps)
np.save(os.path.join(outdir, "consensus_score.npy"), score)
for e in eps:
    print(f"consensus vs e{e}: {spearmanr(score, ep_cons[e]).correlation:.3f}")

DATA = os.environ["MA289_DATA"]
test = pd.read_csv(os.path.join(DATA, "test.csv"))
n = len(score)
p = (pd.Series(score).rank(method="average").to_numpy() - 0.5) / n   # uniform (0,1)
np.save(os.path.join(outdir, "p_rank.npy"), p)
pd.DataFrame({"pair_id": test["pair_id"], "similarity": np.round(p, 6)}).to_csv(
    os.path.join(outdir, "probe_rank.csv"), index=False)
print(f"\nprobe_rank.csv: mean={p.mean():.5f} sd={p.std():.5f} min={p.min():.4f} max={p.max():.4f}")
