"""Replicate the MA289 lesson architecture: Embedding(V,d) -> Linear(d,V), CE loss."""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, pair_cos

class EmbeddingFFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
    def forward(self, token_ids):
        return self.output(self.embedding(token_ids))

def run(d, seed, lr, epochs, batch_size, center, context, V, t1, t2, snapshots):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ci = torch.tensor(center, dtype=torch.long)
    xi = torch.tensor(context, dtype=torch.long)
    loader = DataLoader(TensorDataset(ci, xi), batch_size=batch_size, shuffle=True)
    model = EmbeddingFFNN(V, d)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    out = {}
    for ep in range(1, max(snapshots) + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb); n += len(xb)
        if ep in snapshots:
            E = model.embedding.weight.detach().cpu().numpy().copy()
            out[ep] = dict(cos=pair_cos(E, t1, t2), E=E, loss=tot / n)
    return out

if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    D = load()
    if cfg.get("pairs") == "full":
        import numpy as _np
        a, b = [], []
        for line in D["corpus"]:
            arr = _np.asarray(line)
            n = len(arr)
            for off in (1, 2):
                a.append(arr[:n-off]); b.append(arr[off:])
                a.append(arr[off:]);   b.append(arr[:n-off])
        D["center"] = _np.concatenate(a); D["context"] = _np.concatenate(b)
        print("full pairs:", len(D["center"]), flush=True)
    t0 = time.time()
    res = run(cfg["d"], cfg["seed"], cfg["lr"], max(cfg["snapshots"]), cfg["batch"],
              D["center"], D["context"], D["V"], D["t1"], D["t2"], set(cfg["snapshots"]))
    tag = f"{cfg.get('pairs','sub')}_d{cfg['d']}_s{cfg['seed']}_lr{cfg['lr']}_b{cfg['batch']}"
    for ep, r in res.items():
        np.savez_compressed(os.path.join(cfg["outdir"], f"{tag}_e{ep}.npz"),
                            cos=r["cos"], E=r["E"], loss=r["loss"])
        print(f"{tag}_e{ep} loss={r['loss']:.4f} cos[min/mean/max]="
              f"{r['cos'].min():.3f}/{r['cos'].mean():.3f}/{r['cos'].max():.3f} "
              f"sd={r['cos'].std():.3f} t={time.time()-t0:.0f}s", flush=True)
