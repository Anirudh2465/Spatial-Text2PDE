"""
train_video.py
==============
Training script for VideoMLLM.

Saves checkpoints to:
    checkpoints/video_mllm/best_model.pth
    checkpoints/video_mllm/epoch_<N>.pth

Does NOT touch:
    checkpoints/mllm_*           (PhysicsMLLM)
    checkpoints/image_mllm/      (ImageMLLM)
    checkpoints/grassmann_fno/   (FNO)

Loss
----
  CrossEntropy(text) with numeric tokens upweighted by `--numeric_lambda`
  (default 5.0, same as image_mllm training).
  No separate regression head — Re is encoded in the generated text.

Usage examples
--------------
    python -m src.video_mllm.train_video                      # full run
    python -m src.video_mllm.train_video --dry_run            # 5-batch smoke test
    python -m src.video_mllm.train_video --resume checkpoints/video_mllm/epoch_5.pth
"""

import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.video_mllm.dataset_video import VideoCylinderDataset
from src.video_mllm.model_video   import VideoMLLM

# ── Default paths (match existing training scripts) ────────────────────────────
DATA_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
STAT_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
SAVE_DIR       = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/video_mllm"


# ── Collate ────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    images        = torch.stack([b["image"]        for b in batch])   # (B, C, T, H, W)
    input_ids     = [b["input_ids"]     for b in batch]
    labels        = [b["labels"]        for b in batch]
    numeric_masks = [b["numeric_mask"]  for b in batch]

    input_ids_pad  = pad_sequence(input_ids,     batch_first=True, padding_value=0)
    labels_pad     = pad_sequence(labels,        batch_first=True, padding_value=-100)
    mask_pad       = pad_sequence(numeric_masks, batch_first=True, padding_value=0.0)

    return {
        "image":        images,
        "input_ids":    input_ids_pad,
        "labels":       labels_pad,
        "numeric_mask": mask_pad,
    }


# ── Training loop ──────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  VideoMLLM Training")
    print(f"  Device       : {device}")
    print(f"  Checkpoint   : {SAVE_DIR}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = VideoCylinderDataset(
        DATA_PATH, TOKENIZER_PATH, STAT_PATH,
        num_frames=24, split="train"
    )
    val_ds = VideoCylinderDataset(
        DATA_PATH, TOKENIZER_PATH, STAT_PATH,
        num_frames=24, split="val"
    )
    print(f"  Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        drop_last   = True,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    vocab_size = train_ds.tokenizer.get_vocab_size()
    model = VideoMLLM(
        vocab_size  = vocab_size,
        vision_dim  = 512,
        llm_dim     = 512,
        num_frames  = 24,
        img_size    = 64,
        patch_size  = 16,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params : {total_params/1e6:.2f}M")

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"  Resumed from : {args.resume}")

    # ── Optimiser & Scaler ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR scheduler: linear warmup for first 2 epochs, then cosine decay
    total_steps   = len(train_loader) * args.epochs // args.accum_steps
    warmup_steps  = len(train_loader) * 2        // args.accum_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # ── CSV log ────────────────────────────────────────────────────────────────
    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Epoch", "Duration(s)", "Train_Loss", "Val_Loss", "LR"]
        )

    best_val_loss = float("inf")
    global_step   = 0

    # ── Epoch loop ──────────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        epoch_loss   = 0.0
        t0           = time.time()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for step, batch in enumerate(pbar):
            video        = batch["image"].to(device)        # (B, C, T, H, W)
            input_ids    = batch["input_ids"].to(device)
            labels       = batch["labels"].to(device)
            numeric_mask = batch["numeric_mask"].to(device)

            autocast_ctx = (
                torch.amp.autocast("cuda")
                if torch.cuda.is_available()
                else torch.autocast("cpu")
            )
            with autocast_ctx:
                _, loss = model(
                    video,
                    input_ids,
                    targets             = labels,
                    numeric_mask        = numeric_mask,
                    numeric_loss_lambda = args.numeric_lambda,
                )

            if loss is None or loss.item() == 0.0:
                continue

            scaled_loss = loss / args.accum_steps
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % args.accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if args.dry_run and step >= 4:
                print("  Dry run done.")
                return

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # ── Validation ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                video        = batch["image"].to(device)
                input_ids    = batch["input_ids"].to(device)
                labels       = batch["labels"].to(device)
                numeric_mask = batch["numeric_mask"].to(device)

                with (torch.amp.autocast("cuda") if torch.cuda.is_available() else torch.autocast("cpu")):
                    _, loss = model(
                        video, input_ids,
                        targets             = labels,
                        numeric_mask        = numeric_mask,
                        numeric_loss_lambda = args.numeric_lambda,
                    )
                if loss is not None and loss.item() != 0.0:
                    val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        duration     = time.time() - t0

        print(
            f"  Epoch {epoch+1:>2} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"Time: {duration:.1f}s"
        )

        # CSV log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch+1,
                f"{duration:.1f}",
                f"{avg_train_loss:.4f}",
                f"{avg_val_loss:.4f}",
                f"{scheduler.get_last_lr()[0]:.2e}",
            ])

        # Save latest
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pth"))
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last.pth"))

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  [BEST] New best val loss: {best_val_loss:.4f}")

        # Generation sample after each epoch
        with torch.no_grad():
            sample    = val_ds[0]
            vid_input = sample["image"].unsqueeze(0).to(device)   # (1, C, T, H, W)
            sos       = val_ds.tokenizer.token_to_id("[SOS]")
            start_ids = torch.tensor([[sos]], device=device)
            gen_ids   = model.generate(vid_input, start_ids, max_new_tokens=60)
            gen_text  = val_ds.tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
            print(f"  Sample gen: {gen_text[:120]}")

    print(f"\n  Training complete. Checkpoints saved to: {SAVE_DIR}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoMLLM")
    parser.add_argument("--batch_size",     type=int,   default=4,
                        help="Batch size (default 4; reduce if OOM)")
    parser.add_argument("--epochs",         type=int,   default=20,
                        help="Number of epochs")
    parser.add_argument("--lr",             type=float, default=2e-4,
                        help="Peak learning rate (cosine schedule with warmup)")
    parser.add_argument("--accum_steps",    type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × accum_steps)")
    parser.add_argument("--numeric_lambda", type=float, default=5.0,
                        help="Extra loss weight on numeric/physics tokens")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Run 5 batches then exit (smoke test)")
    args = parser.parse_args()
    train(args)
