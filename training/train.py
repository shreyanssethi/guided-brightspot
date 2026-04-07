"""
training/train.py

Trains either the BaselineUNet or GuidedUNet for WMH segmentation.

Usage:
    # Train baseline (FLAIR + T1, no guidance)
    python training/train.py --model baseline

    # Train guided (FLAIR + T1 + soft map guidance)
    python training/train.py --model guided

    # Resume from checkpoint
    python training/train.py --model guided --resume outputs/checkpoints/guided_best.pt

    # Custom hyperparameters
    python training/train.py --model baseline --epochs 300 --lr 1e-3 --batch_size 4

Key design choices (matched to HW5 + WMH literature):
    Loss:      DiceLoss(to_onehot_y=True, softmax=True)  — HW5 default
    Optimizer: Adam, lr=1e-4                              — HW5 default
    Scheduler: ReduceLROnPlateau (patience=20)            — WMH literature
    Val:       sliding_window_inference every val_interval epochs — HW5 pattern
    Metric:    MONAI DiceMetric (background excluded)     — HW5 default
    Patch:     batch already cropped by dataloader, no extra patching needed

References:
    - HW5 MONAI notebook (course) — training loop structure
    - Li et al. 2018 (WMH challenge winner) — 300-600 epochs, Adam 1e-4
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism

# ── Repo imports ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.dataset import get_dataloaders, PATCH_SIZE
from training.models import build_baseline, build_guided


# ── Defaults ───────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = REPO_ROOT / 'outputs' / 'checkpoints'
LOG_DIR        = REPO_ROOT / 'outputs' / 'logs'
VAL_INTERVAL   = 5      # validate every N epochs
SW_BATCH_SIZE  = 2      # sliding window inference batch size


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Train baseline or guided U-Net for WMH segmentation'
    )
    p.add_argument(
        '--model', choices=['baseline', 'guided'], required=True,
        help='Which model to train: baseline (FLAIR+T1) or guided (+ soft map)'
    )
    p.add_argument(
        '--gpu', type=int, default=0,
        help='GPU index to use (default: 0)'
    )
    p.add_argument(
        '--epochs', type=int, default=300,
        help='Number of training epochs (default: 300)'
    )
    p.add_argument(
        '--lr', type=float, default=1e-4,
        help='Initial learning rate (default: 1e-4, matched to HW5)'
    )
    p.add_argument(
        '--batch_size', type=int, default=2,
        help='Batch size (default: 2). Each sample produces 4 patches → 8 patches/step'
    )
    p.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes (default: 4)'
    )
    p.add_argument(
        '--cache_rate', type=float, default=1.0,
        help='MONAI CacheDataset cache rate (default: 1.0 = cache all in RAM)'
    )
    p.add_argument(
        '--val_interval', type=int, default=VAL_INTERVAL,
        help='Validate every N epochs (default: 5)'
    )
    p.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from'
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    return p.parse_args()


# ── Training utilities ─────────────────────────────────────────────────────────

def get_input(batch, model_type: str, device: torch.device):
    """
    Build model input tensor(s) from a dataloader batch.

    For baseline: concatenate FLAIR + T1 → (B, 2, H, W, D)
    For guided:   same concat + return soft_map → (B, 1, H, W, D)

    Returns:
        x        (B, 2, H, W, D)
        soft_map (B, 1, H, W, D) or None
        labels   (B, 1, H, W, D)
    """
    flair    = batch['flair'].to(device)
    t1       = batch['t1'].to(device)
    labels   = batch['wmh'].to(device)
    x        = torch.cat([flair, t1], dim=1)

    if model_type == 'guided':
        soft_map = batch['soft_map'].to(device)
    else:
        soft_map = None

    return x, soft_map, labels


def forward(model, x, soft_map, model_type: str):
    """Single forward pass, handles both model signatures."""
    if model_type == 'guided':
        return model(x, soft_map)
    return model(x)


def save_checkpoint(path: Path, model, optimizer, scheduler,
                    epoch: int, best_metric: float, history: dict):
    """Save full training state to disk."""
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric':          best_metric,
        'history':              history,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    """Load training state from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch  = ckpt['epoch'] + 1
    best_metric  = ckpt['best_metric']
    history      = ckpt.get('history', {'train_loss': [], 'val_dice': []})
    print(f'Resumed from epoch {ckpt["epoch"]}  (best val DICE so far: {best_metric:.4f})')
    return start_epoch, best_metric, history


def save_log(log_path: Path, history: dict):
    """Save training history as JSON (easy to load for plotting later)."""
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────────
    set_determinism(seed=args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}  (GPU {args.gpu})')

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = CHECKPOINT_DIR / f'{args.model}_best.pt'
    last_ckpt_path = CHECKPOINT_DIR / f'{args.model}_last.pt'
    log_path       = LOG_DIR        / f'{args.model}_history.json'

    # ── Data ───────────────────────────────────────────────────────────────────
    print('\nBuilding dataloaders...')
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        seed=args.seed,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f'\nBuilding {args.model} model...')
    if args.model == 'baseline':
        model = build_baseline().to(device)
    else:
        model = build_guided().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params:,}')

    # ── Loss, optimizer, scheduler ─────────────────────────────────────────────
    # DiceLoss with softmax + one-hot — matched to HW5 exactly
    loss_fn   = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Reduce LR when val DICE plateaus — standard for WMH segmentation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20,
    )

    # ── Validation utilities (matched to HW5) ──────────────────────────────────
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    post_pred   = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label  = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # ── Resume if requested ────────────────────────────────────────────────────
    start_epoch = 0
    best_metric = -1.0
    history     = {'train_loss': [], 'val_dice': [], 'lr': []}

    if args.resume:
        start_epoch, best_metric, history = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f'\nTraining {args.model} for {args.epochs} epochs '
          f'(val every {args.val_interval})\n')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        # ── Train epoch ──────────────────────────────────────────────────────
        pbar = tqdm(train_loader,
                    desc=f'Epoch {epoch+1:3d}/{args.epochs} [train]',
                    unit='batch', leave=False)

        for batch in pbar:
            x, soft_map, labels = get_input(batch, args.model, device)

            optimizer.zero_grad()
            outputs = forward(model, x, soft_map, args.model)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        mean_loss = epoch_loss / n_steps
        history['train_loss'].append(round(mean_loss, 5))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f'Epoch {epoch+1:3d}/{args.epochs}  '
              f'loss={mean_loss:.4f}  '
              f'lr={optimizer.param_groups[0]["lr"]:.2e}', end='')

        # ── Validation ───────────────────────────────────────────────────────
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(val_loader,
                                desc=f'Epoch {epoch+1:3d}/{args.epochs} [val]',
                                unit='vol', leave=False)

                for val_batch in val_pbar:
                    x_val, sm_val, labels_val = get_input(
                        val_batch, args.model, device
                    )

                    # sliding_window_inference for full-volume validation
                    # Wraps the forward call so it works with both model types
                    if args.model == 'guided':
                        def _forward_fn(x_patch):
                            # Resize soft map to patch spatial dims on the fly
                            sm_patch = F.interpolate(
                                sm_val,
                                size=x_patch.shape[2:],
                                mode='trilinear',
                                align_corners=False,
                            )
                            return model(x_patch, sm_patch)
                    else:
                        def _forward_fn(x_patch):
                            return model(x_patch)

                    val_outputs = sliding_window_inference(
                        x_val, PATCH_SIZE, SW_BATCH_SIZE, _forward_fn
                    )

                    val_outputs = [post_pred(i)
                                   for i in decollate_batch(val_outputs)]
                    labels_val  = [post_label(i)
                                   for i in decollate_batch(labels_val)]
                    dice_metric(y_pred=val_outputs, y=labels_val)

            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            history['val_dice'].append(round(val_dice, 5))
            scheduler.step(val_dice)

            print(f'  val_dice={val_dice:.4f}', end='')

            # Save best checkpoint
            if val_dice > best_metric:
                best_metric      = val_dice
                best_metric_epoch = epoch + 1
                save_checkpoint(
                    best_ckpt_path, model, optimizer, scheduler,
                    epoch, best_metric, history
                )
                print(f'  ← new best, saved to {best_ckpt_path.name}', end='')

        print()  # newline after each epoch's summary

        # Save last checkpoint every 10 epochs for safety
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                last_ckpt_path, model, optimizer, scheduler,
                epoch, best_metric, history
            )
            save_log(log_path, history)

    # ── Final save ─────────────────────────────────────────────────────────────
    save_checkpoint(
        last_ckpt_path, model, optimizer, scheduler,
        args.epochs - 1, best_metric, history
    )
    save_log(log_path, history)

    print(f'\nTraining complete.')
    print(f'Best val DICE: {best_metric:.4f}')
    print(f'Checkpoint:    {best_ckpt_path}')
    print(f'Log:           {log_path}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()

    print('=' * 60)
    print(f'  BrightSpot WMH Training')
    print(f'  Model:   {args.model}')
    print(f'  Epochs:  {args.epochs}')
    print(f'  LR:      {args.lr}')
    print(f'  Batch:   {args.batch_size}')
    print(f'  GPU:     {args.gpu}')
    print(f'  Resume:  {args.resume or "none"}')
    print('=' * 60)

    train(args)