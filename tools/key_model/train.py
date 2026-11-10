"""Train and evaluate the key-detection model (development tool).

    python tools/key_model/train.py --train BP.npz BP_LABELS --test GS.npz GS_LABELS
        [--also-train-on-test] [--out stemdeck/analysis/key_model.npz]

Model
-----
A small neural network scores "does this note-energy picture look like a
major key / a minor key whose home note is C?". To score all 24 keys the
picture is slid up or down so that each candidate home note in turn lands on
C, and the same network is applied. Sharing one network across all twelve
transpositions means every training track teaches the model about every
key, which matters with only ~1,400 labelled tracks.

The exported file holds plain weight matrices; the app evaluates them with
numpy (see stemdeck/analysis/key.py), so no training code ships in the app.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import key_index, load_features, load_labels, summarize, track_id  # noqa: E402

from stemdeck.analysis.keyfeatures import window_features  # noqa: E402

#: Slide amount (semitones) that brings pitch class k onto C, staying within +/-6.
SHIFTS = [-k if k <= 6 else 12 - k for k in range(12)]


def build_dataset(feats, labels, window, hop):
    xs, ys, groups = [], [], []
    names = [n for n in sorted(feats) if track_id(n) in labels]
    for g, name in enumerate(names):
        w = window_features(feats[name], window, hop)
        if len(w) == 0:
            continue
        xs.append(w)
        ys.extend([key_index(*labels[track_id(name)])] * len(w))
        groups.extend([g] * len(w))
    return np.concatenate(xs), np.asarray(ys), np.asarray(groups), names


def shift_bank(x: torch.Tensor) -> torch.Tensor:
    """(B, 72) -> (B, 12, 72): row k has pitch class k moved onto C (zero filled)."""
    out = x.new_zeros(x.shape[0], 12, x.shape[1])
    for k, s in enumerate(SHIFTS):
        if s == 0:
            out[:, k] = x
        elif s > 0:
            out[:, k, s:] = x[:, :-s]
        else:
            out[:, k, :s] = x[:, -s:]
    return out


class KeyNet(nn.Module):
    def __init__(self, n_in=72, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x: (B, 72) window features
        scores = self.net(shift_bank(x))        # (B, 12, 2)
        return torch.cat([scores[:, :, 0], scores[:, :, 1]], dim=1)  # (B, 24)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Mild, music-preserving variation: tonal balance tilt and noise."""
    b, n = x.shape
    pos = torch.linspace(-1, 1, n)
    tilt = (torch.rand(b, 1, generator=gen) - 0.5) * 1.0 * pos
    bump = (torch.rand(b, 1, generator=gen) - 0.5) * 0.6 * torch.cos(
        np.pi * (pos[None] + torch.rand(b, 1, generator=gen)))
    noise = 0.05 * torch.randn(b, n, generator=gen)
    return torch.clamp(x + tilt + bump + noise, min=0.0)


def predict_tracks(model, x, groups):
    model.eval()
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.from_numpy(x)), dim=1).numpy()
    preds = {}
    for g in np.unique(groups):
        preds[int(g)] = int(logp[groups == g].mean(axis=0).argmax())
    return preds


def train_model(x, y, epochs, seed, hidden, dropout, lr=2e-3, wd=1e-3, log=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = KeyNet(x.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(x) + 255) // 256))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = loss_fn(model(augment(xt[idx], gen)), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if log and (epoch + 1) % 20 == 0:
            log(epoch + 1, total / len(perm), model)
    return model


def export(model: KeyNet, path: str, meta: dict):
    arrays = {}
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        arrays[f"w{i}"] = lin.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32)
    arrays["shifts"] = np.asarray(SHIFTS, dtype=np.int64)
    for k, v in meta.items():
        arrays[f"meta_{k}"] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs=2, required=True, metavar=("FEATS", "LABELS"))
    ap.add_argument("--test", nargs=2, required=True, metavar=("FEATS", "LABELS"))
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--hidden", default="256,128")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--also-train-on-test", action="store_true",
                    help="after reporting the held-out score, retrain on everything for shipping")
    ap.add_argument("--out")
    args = ap.parse_args()
    hidden = tuple(int(h) for h in args.hidden.split(","))

    xtr, ytr, gtr, _ = build_dataset(load_features(args.train[0]), load_labels(args.train[1]),
                                     args.window, args.hop)
    xte, yte, gte, _ = build_dataset(load_features(args.test[0]), load_labels(args.test[1]),
                                     args.window, args.hop)
    true_te = {int(g): int(yte[gte == g][0]) for g in np.unique(gte)}
    print(f"train: {len(np.unique(gtr))} tracks / {len(xtr)} windows; "
          f"test: {len(true_te)} tracks / {len(xte)} windows")

    def report(epoch, loss, model):
        p = predict_tracks(model, xte, gte)
        s = summarize([true_te[g] for g in p], [p[g] for g in p])
        print(f"  epoch {epoch:4d} loss {loss:.3f}  test correct {s['correct']} weighted {s['weighted']}")

    scores = []
    for seed in range(args.seeds):
        model = train_model(xtr, ytr, args.epochs, seed, hidden, args.dropout,
                            log=report if seed == 0 else None)
        p = predict_tracks(model, xte, gte)
        s = summarize([true_te[g] for g in p], [p[g] for g in p])
        scores.append(s)
        print(f"seed {seed}: HELD-OUT {s}")
    if len(scores) > 1:
        print("mean correct", np.mean([s["correct"] for s in scores]).round(1),
              "mean weighted", np.mean([s["weighted"] for s in scores]).round(1))

    if args.out:
        if args.also_train_on_test:
            x = np.concatenate([xtr, xte])
            y = np.concatenate([ytr, yte])
            model = train_model(x, y, args.epochs, 0, hidden, args.dropout)
            print(f"retrained on train+test ({len(x)} windows) for shipping")
        export(model, args.out, {
            "window": args.window, "hop": args.hop,
            "heldout_correct": scores[0]["correct"], "heldout_weighted": scores[0]["weighted"],
        })
        print("saved", args.out)


if __name__ == "__main__":
    main()
