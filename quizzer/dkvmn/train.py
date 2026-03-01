# dkvmn/train.py
from __future__ import annotations

import json
import random
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

from dkvmn.dataset import load_skills, build_skill2idx, load_sequences, DKVMNDataset, collate_pad
from dkvmn.model import DKVMN


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_val(seqs: List[dict], val_frac: float, seed: int) -> Tuple[List[dict], List[dict]]:
    seqs = list(seqs)
    random.Random(seed).shuffle(seqs)
    cut = int((1.0 - val_frac) * len(seqs))
    cut = max(1, min(cut, len(seqs) - 1))  # ensure both sides non-empty
    return seqs[:cut], seqs[cut:]


@torch.no_grad()
def evaluate(model: DKVMN, dl: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_targets = []

    for skill, corr, mask in dl:
        skill = skill.to(device)
        corr = corr.to(device)
        mask = mask.to(device)

        pred_masked, targ_masked = model(skill, corr, mask)  # both already * mask

        pred_flat = pred_masked.view(-1)
        targ_flat = targ_masked.view(-1)
        mask_flat = mask.view(-1)

        idx = mask_flat > 0.5
        if idx.sum().item() == 0:
            continue

        p = pred_flat[idx].detach().cpu().numpy()
        y = targ_flat[idx].detach().cpu().numpy()

        all_probs.append(p)
        all_targets.append(y)

    if not all_probs:
        return {"val_auc": None, "val_logloss": None, "n": 0}

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0).astype(int)

    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    if len(np.unique(targets)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(targets, probs))

    ll = float(log_loss(targets, probs, labels=[0, 1]))
    return {"val_auc": auc, "val_logloss": ll, "n": int(targets.shape[0])}


def train(
    skills_path: str = "data/skills.json",
    interactions_path: str = "data/interactions.jsonl",
    out_model_path: str = "dkvmn_model.pt",
    out_skill_map_path: str = "skill2idx.json",
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_len: int = 200,
    min_len: int = 2,
    memory_size: int = 50,
    d_k: int = 64,
    d_v: int = 64,
    val_frac: float = 0.1,
    seed: int = 42,
    weight_decay: float = 1e-4,
    patience: Optional[int] = 4,
    sort_by_time: bool = True,
):
    _set_seed(seed)

    # 1) Build skill map
    skills = load_skills(skills_path)
    skill2idx = build_skill2idx(skills)

    # 2) Load sequences (auto-detect seq-style vs event-log and normalize)
    seqs = load_sequences(interactions_path, sort_by_time=sort_by_time)
    if len(seqs) < 2:
        raise RuntimeError(f"Not enough student sequences loaded from {interactions_path} (got {len(seqs)})")

    train_seqs, val_seqs = _split_train_val(seqs, val_frac=val_frac, seed=seed)

    train_ds = DKVMNDataset(train_seqs, skill2idx, max_len=max_len, min_len=min_len)
    val_ds = DKVMNDataset(val_seqs, skill2idx, max_len=max_len, min_len=min_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    print(f"Loaded student sequences: total={len(seqs)} train={len(train_seqs)} val={len(val_seqs)}")
    print(f"Dataset samples: train={len(train_ds)} val={len(val_ds)} (min_len={min_len}, max_len={max_len})")
    print(f"Num skills: {len(skill2idx)}")

    # 3) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DKVMN(n_skills=len(skill2idx), memory_size=memory_size, d_k=d_k, d_v=d_v).to(device)

    # 4) Optimizer + loss
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce_sum = torch.nn.BCELoss(reduction="sum")

    best_score = -1e9
    best_ep = 0
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0.0

        for skill, corr, mask in tqdm(train_dl, desc=f"epoch {ep}/{epochs}"):
            skill = skill.to(device)
            corr = corr.to(device)
            mask = mask.to(device)

            opt.zero_grad()
            pred_masked, targ_masked = model(skill, corr, mask)

            pred_flat = pred_masked.view(-1)
            targ_flat = targ_masked.view(-1)
            mask_flat = mask.view(-1)

            idx = mask_flat > 0.5
            pred_valid = pred_flat[idx]
            targ_valid = targ_flat[idx]

            if pred_valid.numel() == 0:
                continue

            loss = bce_sum(pred_valid, targ_valid) / (pred_valid.numel() + 1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item() * pred_valid.numel()
            total_count += pred_valid.numel()

        train_loss = total_loss / (total_count + 1e-9)

        metrics = evaluate(model, val_dl, device=device)
        val_auc = metrics["val_auc"]
        val_ll = metrics["val_logloss"]

        if val_auc is None:
            print(f"Epoch {ep}: train_loss={train_loss:.4f} | val_auc=N/A | val_logloss={val_ll:.4f} n={metrics['n']}")
            score = -val_ll if val_ll is not None else -1e9
        else:
            print(f"Epoch {ep}: train_loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_logloss={val_ll:.4f} n={metrics['n']}")
            score = val_auc

        improved = score > best_score + 1e-6
        if improved:
            best_score = score
            best_ep = ep
            bad_epochs = 0

            torch.save({"state_dict": model.state_dict(), "n_skills": len(skill2idx)}, out_model_path)
            with open(out_skill_map_path, "w") as f:
                json.dump(skill2idx, f, indent=2)

            print(f"  ✓ Saved best model @ epoch {ep} -> {out_model_path}")
        else:
            bad_epochs += 1

        if patience is not None and bad_epochs >= patience:
            print(f"Early stopping: no improvement for {patience} epochs. Best epoch was {best_ep}.")
            break

    print(f"Done. Best epoch: {best_ep}. Best score (AUC or -logloss): {best_score:.4f}")
    print(f"Best model saved -> {out_model_path}")
    print(f"Skill map saved -> {out_skill_map_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--skills_path", default="data/skills.json")
    p.add_argument("--interactions_path", default="data/interactions.jsonl")
    p.add_argument("--out_model_path", default="dkvmn_model.pt")
    p.add_argument("--out_skill_map_path", default="skill2idx.json")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--min_len", type=int, default=2)

    p.add_argument("--memory_size", type=int, default=50)
    p.add_argument("--d_k", type=int, default=64)
    p.add_argument("--d_v", type=int, default=64)

    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--no_sort_by_time", action="store_true")

    args = p.parse_args()

    train(
        skills_path=args.skills_path,
        interactions_path=args.interactions_path,
        out_model_path=args.out_model_path,
        out_skill_map_path=args.out_skill_map_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_len=args.max_len,
        min_len=args.min_len,
        memory_size=args.memory_size,
        d_k=args.d_k,
        d_v=args.d_v,
        val_frac=args.val_frac,
        seed=args.seed,
        weight_decay=args.weight_decay,
        patience=args.patience,
        sort_by_time=(not args.no_sort_by_time),
    )