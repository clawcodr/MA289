import os, sys, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load
D = load()
freq = collections.Counter(t for line in D["corpus"] for t in line)
V = D["V"]
order = [t for t, _ in freq.most_common()]
rank = {t: i for i, t in enumerate(order)}
tt = sorted(set(D["t1"]).union(D["t2"]))
print("distinct test tokens:", len(tt))
rs = sorted(rank[t] for t in tt)
print("freq-rank of test tokens: min", rs[0], "max", rs[-1], "median", rs[len(rs)//2])
print("is it exactly the top-N most frequent?", rs == list(range(len(rs))))
cnt = np.array([freq[t] for t in tt])
print("test-token counts: min", cnt.min(), "max", cnt.max(), "median", int(np.median(cnt)))
allc = np.array([freq[t] for t in range(V)])
print("all-token counts:  min", allc.min(), "max", allc.max(), "median", int(np.median(allc)))
# duplicate pairs / self pairs?
pairs = list(zip(D["t1"].tolist(), D["t2"].tolist()))
print("self-pairs:", sum(1 for a, b in pairs if a == b),
      "| duplicate unordered pairs:", len(pairs) - len({frozenset(p) for p in pairs}))
# how often do test pairs co-occur in window 2?
co = collections.Counter()
for line in D["corpus"]:
    a = np.asarray(line); n = len(a)
    for off in (1, 2):
        for x, y in zip(a[:n-off], a[off:]):
            co[(x, y)] += 1; co[(y, x)] += 1
z = np.array([co.get(p, 0) for p in pairs])
print("test pairs with zero window-2 co-occurrence:", int((z == 0).sum()), "/", len(z))
