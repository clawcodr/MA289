import os, sys, glob, numpy as np
from scipy.stats import spearmanr
d = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.npz")):
    b = os.path.basename(f)[:-4]
    seed = b.split("_s")[1].split("_")[0]; ep = int(b.split("_e")[-1])
    d.setdefault(ep, {})[seed] = np.load(f)["cos"]
print("epoch  n  mean seed-seed r   implied corr(single run, infinite-seed consensus)")
for ep in sorted(d):
    cs = list(d[ep].values())
    rs = [spearmanr(cs[i], cs[j]).correlation
          for i in range(len(cs)) for j in range(i+1, len(cs))]
    m = np.mean(rs)
    print(f"  {ep:3d}  {len(cs)}      {m:.3f}            {np.sqrt(max(m,0)):.3f}")
eps = sorted(d)
print("\nepoch-to-epoch (same seed, averaged):")
print("        " + " ".join(f"{e:>6d}" for e in eps))
for e1 in eps:
    print(f"  e{e1:<5d} " + " ".join(
        f"{np.mean([spearmanr(d[e1][s], d[e2][s]).correlation for s in d[e1] if s in d[e2]]):6.3f}"
        for e2 in eps))
