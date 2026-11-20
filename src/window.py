import os, sys, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load
D = load(); V = D["V"]
corpus = D["corpus"]
obs = collections.Counter(zip(D["center"].tolist(), D["context"].tolist()))
print("distinct observed pairs:", len(obs), "total:", sum(obs.values()))
# co-occurrence counts for each window size, and coverage of observed pairs
for w in (1, 2, 3, 5):
    cnt = collections.Counter()
    for line in corpus:
        n = len(line)
        for i, c in enumerate(line):
            for j in range(max(0, i - w), min(n, i + w + 1)):
                if j != i:
                    cnt[(c, line[j])] += 1
    covered = sum(1 for p in obs if p in cnt)
    print(f"w={w}: distinct cooc {len(cnt)}, total {sum(cnt.values())}, "
          f"observed pairs covered {covered}/{len(obs)} = {covered/len(obs):.4f}")
