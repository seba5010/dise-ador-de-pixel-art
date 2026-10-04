import torch
from pathlib import Path

exclude = {"generator", "discriminator", "opt_g", "opt_d", "scaler_g", "scaler_d", "rng_state", "cuda_rng_state", "scheduler_g", "scheduler_d"}

for f in ['checkpoints/best_generator.pt', 'checkpoints/latest_checkpoint.pt']:
    p = Path(f)
    if not p.exists():
        print(f"{f}: NOT FOUND")
        continue
    ckpt = torch.load(p, map_location='cpu')
    if isinstance(ckpt, dict):
        ep = ckpt.get('epoch', '?')
        loss = ckpt.get('loss', ckpt.get('g_loss', '?'))
        best = ckpt.get('best_loss', '?')
        keys = sorted([k for k in ckpt.keys() if k not in exclude])
        print(f"{f}: epoch={ep}, loss={loss}, best_loss={best}")
        print(f"  keys: {keys}")
    else:
        print(f"{f}: not dict, type={type(ckpt)}")

snap_dir = Path('checkpoints/snapshots')
if snap_dir.exists():
    snaps = sorted(snap_dir.glob('checkpoint_epoch_*.pt'))
    print(f"\nSnapshots found: {len(snaps)}")
    for s in snaps:
        ckpt = torch.load(s, map_location='cpu')
        if isinstance(ckpt, dict):
            ep = ckpt.get('epoch', '?')
            bl = ckpt.get('best_loss', '?')
            print(f"  {s.name}: epoch={ep}, best_loss={bl}")
        else:
            print(f"  {s.name}: not dict")
else:
    print("\nNo snapshots directory found")
