"""
train_grassmann_ae.py — Training script for GrassmannFNOAutoencoder

Usage:
    python scripts/06_grassmann_fno/train_grassmann_ae.py \
        --data_path path/to/train_downsampled_labeled.h5 \
        --stat_path path/to/train_normal_stat.pkl \
        --epochs 200 --rank 16 --grid_size 16 --batch_size 2

All hyperparameters can be overridden via CLI. Run with --help for details.
"""

import os
import sys
import math
import argparse
import logging
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from src.models.grassmann_fno_ae import GrassmannFNOAutoencoder
from src.data.mesh_dataset import CylinderMeshDataset, PaddedMeshCollator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train Grassmann-FNO Autoencoder')

    # Data
    p.add_argument('--data_path', type=str,
                   default='d:/Semester 6/Natural Language Processing/Project 3/train_downsampled_labeled.h5')
    p.add_argument('--stat_path', type=str,
                   default='d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl')
    p.add_argument('--save_dir', type=str,
                   default='d:/Semester 6/Natural Language Processing/Project 3/checkpoints/grassmann_fno')
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--num_timesteps', type=int, default=25)
    p.add_argument('--val_split', type=float, default=0.1,
                   help='Fraction of data for validation')

    # Architecture
    p.add_argument('--rank',            type=int,   default=16,  help='Grassmann rank k')
    p.add_argument('--grid_size',       type=int,   default=16,  help='FNO grid size G')
    p.add_argument('--fno_modes',       type=int,   default=8,   help='Fourier modes per dim')
    p.add_argument('--fno_hidden',      type=int,   default=64)
    p.add_argument('--fno_layers',      type=int,   default=4)
    p.add_argument('--hidden_channels', type=int,   default=64)
    p.add_argument('--z_channels',      type=int,   default=16)
    p.add_argument('--dropout',         type=float, default=0.1)

    # Training
    p.add_argument('--epochs',     type=int,   default=200)
    p.add_argument('--batch_size', type=int,   default=2)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--kl_weight',  type=float, default=1e-6)
    p.add_argument('--grad_clip',  type=float, default=1.0)
    p.add_argument('--save_every', type=int,   default=25,  help='Save checkpoint every N epochs')
    p.add_argument('--resume',     type=str,   default=None, help='Path to checkpoint to resume from')
    p.add_argument('--device',     type=str,   default='auto')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_device(args) -> torch.device:
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(args.device)


def make_model(args) -> GrassmannFNOAutoencoder:
    modes = (args.fno_modes, args.fno_modes, args.fno_modes)
    return GrassmannFNOAutoencoder(
        in_channels=3,
        rank=args.rank,
        grid_size=args.grid_size,
        nt=args.num_timesteps,
        fno_modes=modes,
        fno_hidden=args.fno_hidden,
        fno_layers=args.fno_layers,
        hidden_channels=args.hidden_channels,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        dropout=args.dropout,
        z_channels=args.z_channels,
        kl_weight=args.kl_weight,
    )


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, path)
    log.info(f'Checkpoint saved → {path}')


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and ckpt.get('scheduler_state_dict'):
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    log.info(f'Resumed from epoch {ckpt["epoch"]} (loss={ckpt["loss"]:.6f})')
    return start_epoch


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = get_device(args)
    log.info(f'Device: {device}')

    # ---- Dataset -----------------------------------------------------------
    log.info(f'Loading dataset: {args.data_path}')
    dataset = CylinderMeshDataset(
        file_path=args.data_path,
        num_timesteps=args.num_timesteps,
        normalise=True,
        stat_path=args.stat_path,
        max_samples=args.max_samples,
    )
    log.info(f'Dataset size: {len(dataset)} samples')

    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    collate = PaddedMeshCollator()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False)

    # ---- Model & optimiser -------------------------------------------------
    model = make_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Model parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)

    # ---- Training ----------------------------------------------------------
    best_val_loss = float('inf')
    history = {'train_recon': [], 'train_kl': [], 'val_recon': []}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # --- Train
        model.train()
        tr_recon = tr_kl = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)   # (B, T, M, 3)

            optimizer.zero_grad()
            _, loss_dict = model(batch_x)
            loss_dict['total'].backward()

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            tr_recon += loss_dict['recon'].item()
            tr_kl    += loss_dict['kl'].item()

        scheduler.step()
        tr_recon /= len(train_loader)
        tr_kl    /= len(train_loader)

        # --- Validate
        model.eval()
        val_recon = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                _, loss_dict = model(batch_x)
                val_recon += loss_dict['recon'].item()
        val_recon /= len(val_loader)

        elapsed = time.time() - t0
        log.info(f'Epoch {epoch+1:4d}/{args.epochs} | '
                 f'train_recon={tr_recon:.5f}  kl={tr_kl:.5f} | '
                 f'val_recon={val_recon:.5f} | '
                 f'lr={scheduler.get_last_lr()[0]:.2e} | '
                 f't={elapsed:.1f}s')

        history['train_recon'].append(tr_recon)
        history['train_kl'].append(tr_kl)
        history['val_recon'].append(val_recon)

        # --- Checkpointing
        if val_recon < best_val_loss:
            best_val_loss = val_recon
            save_checkpoint(model, optimizer, scheduler, epoch, val_recon,
                            os.path.join(args.save_dir, 'best.pth'))

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_recon,
                            os.path.join(args.save_dir, f'epoch_{epoch+1:04d}.pth'))

    log.info(f'Training complete. Best val_recon={best_val_loss:.6f}')

    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_val_loss,
                    os.path.join(args.save_dir, 'final.pth'))


if __name__ == '__main__':
    main()
