"""
evaluate_models.py
==================
Side-by-side comparison of:
  - PhysicsMLLM  (src/mllm)        : Video ViT + Grassmann Projector + GPT + Regression Head
  - ImageMLLM    (src/image_mllm)  : Image ViT + MLP Projector + GPT (numeric-upweighted loss)

Usage:
    python evaluate_models.py                         # random sample
    python evaluate_models.py --idx 42               # specific dataset index
    python evaluate_models.py --n_samples 5          # evaluate N random samples and summarise
    python evaluate_models.py --mllm_ckpt <path>     # override checkpoint paths
    python evaluate_models.py --img_ckpt  <path>
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import re
import random
import argparse
import textwrap

import torch
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tokenizers import Tokenizer

# ── make sure project root is on sys.path ──────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from src.mllm.model import PhysicsMLLM
from src.mllm.dataset import PhysicsMLLMDataset
from src.image_mllm.model_image import ImageMLLM
from src.image_mllm.dataset_image import ImageCylinderDataset
from src.image_mllm.metrics import PLCSScorer

# ── Default Paths (match training scripts) ─────────────────────────────────────
DATA_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
STAT_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
MLLM_CKPT      = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/mllm_last.pth"
IMG_CKPT       = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/image_mllm/best_model.pth"

SEPARATOR = "=" * 72


# ── Physics extraction (mirrors PLCSScorer but standalone) ────────────────────
def extract_physics(text: str) -> dict:
    """Regex-extract numeric physics quantities from a generated caption."""
    m_re  = re.search(r"(?:Reynolds number\s*is|Reynolds number\s*of|Re\s*=|Re=)\s*([\d\.]+)", text)
    m_rad = re.search(r"radius(?:\s*of|\s*:)?\s*([\d\.]+)", text)
    m_vel = re.search(r"velocity(?:\s*of|\s*is|\s*at the inlet is|\s*=)?\s*([\d\.]+)", text)
    m_pos = re.search(r"(?:position:|located at \(|at\s*\(?|X=)\s*([\d\.]+)(?:,\s*Y=|,\s*)([\d\.]+)", text)
    m_flow = re.search(r"(laminar|unsteady laminar|turbulent transition|turbulent)", text, re.IGNORECASE)

    def _f(m, g=1):
        """Safe float: strip trailing periods/commas before conversion."""
        try:
            return float(m.group(g).strip(".,")) if m else None
        except (ValueError, AttributeError):
            return None

    return {
        "re":     _f(m_re),
        "radius": _f(m_rad),
        "vel":    _f(m_vel),
        "px":     _f(m_pos, 1),
        "py":     _f(m_pos, 2),
        "flow":   m_flow.group(1).lower() if m_flow else None,
    }


def relative_error(true_val, pred_val) -> float | None:
    """Returns 0-1 accuracy score (1 = perfect, 0 = very wrong or missing)."""
    if true_val is None or pred_val is None:
        return None
    return max(0.0, 1.0 - abs(true_val - pred_val) / (abs(true_val) + 1e-8))


def score_output(true_text: str, pred_text: str, true_re: float) -> dict:
    """
    Compute a Physics Language Composite Score (PLCS) breakdown.
    Falls back gracefully if optional NLP packages (bert_score) are missing.
    """
    true_phys = extract_physics(true_text)
    pred_phys = extract_physics(pred_text)

    # --- Numeric accuracy ---
    re_score   = relative_error(true_re, pred_phys["re"])
    rad_score  = relative_error(true_phys["radius"], pred_phys["radius"])
    vel_score  = relative_error(true_phys["vel"], pred_phys["vel"])
    flow_match = (
        (true_phys["flow"] is not None and pred_phys["flow"] is not None
         and true_phys["flow"] == pred_phys["flow"])
    )

    numeric_scores = [s for s in [re_score, rad_score, vel_score] if s is not None]
    numeric_avg    = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    physical_score = 0.8 * numeric_avg + 0.2 * float(flow_match)

    # --- NLP scores (optional) ---
    nlp_score = None
    bert_f1   = rouge_l = None
    try:
        from bert_score import score as bscore
        from rouge_score import rouge_scorer as rs
        _, _, F1 = bscore([pred_text], [true_text], lang="en",
                          model_type="distilbert-base-uncased", verbose=False)
        bert_f1 = F1.item()
        scorer  = rs.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(true_text, pred_text)["rougeL"].fmeasure
        nlp_score = (bert_f1 + rouge_l) / 2.0
    except Exception:
        pass  # NLP scoring is optional

    plcs = (0.7 * physical_score + 0.3 * nlp_score) if nlp_score is not None else physical_score

    return {
        "plcs":             plcs,
        "physical_score":   physical_score,
        "numeric_avg":      numeric_avg,
        "re_score":         re_score,
        "rad_score":        rad_score,
        "vel_score":        vel_score,
        "flow_match":       flow_match,
        "nlp_score":        nlp_score,
        "bert_f1":          bert_f1,
        "rouge_l":          rouge_l,
        "pred_re":          pred_phys["re"],
        "pred_flow":        pred_phys["flow"],
    }


# ── Model loaders ──────────────────────────────────────────────────────────────
def load_physics_mllm(tokenizer_path, ckpt_path, device):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = PhysicsMLLM(
        vocab_size  = tokenizer.get_vocab_size(),
        vision_dim  = 256,
        llm_dim     = 256,
    ).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  [PhysicsMLLM]  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [PhysicsMLLM]  [WARNING] Checkpoint NOT found: {ckpt_path}  (running with random weights)")
    model.eval()
    return model, tokenizer


def load_image_mllm(tokenizer_path, ckpt_path, device):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = ImageMLLM(
        vocab_size  = tokenizer.get_vocab_size(),
        vision_dim  = 512,
        llm_dim     = 512,
        img_size    = 64,
        patch_size  = 16,
    ).to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  [ImageMLLM]    Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  [ImageMLLM]    [WARNING] Checkpoint NOT found: {ckpt_path}  (running with random weights)")
    model.eval()
    return model, tokenizer


# ── Inference helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def run_physics_mllm(model, tokenizer, video_tensor, device, max_new_tokens=80, temperature=0.7):
    """
    video_tensor: (C, T, H, W)  ← dataset output shape
    Returns (generated_text, predicted_re)
    """
    video = video_tensor.unsqueeze(0).to(device)   # (1, C, T, H, W)

    # Regression prediction
    vis    = model.vision_encoder(video)
    proj   = model.projector(vis)
    re_pred = model.regression_head(proj.mean(dim=1)).item()

    # Text generation
    generated_text = model.generate(video, tokenizer,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature)
    return generated_text, re_pred


@torch.no_grad()
def run_image_mllm(model, tokenizer, video_or_frame, device, max_new_tokens=80, temperature=0.7):
    """
    Accepts either:
      - a single frame  (C, H, W)
      - a full video    (C, T, H, W)

    For video input: runs ImageViT on each frame independently, then
    mean-pools the patch embeddings across the time dimension before
    passing to the projector + ImageGPT decoder. This gives the model
    temporal context without requiring any weight changes.

    Returns: generated_text string
    """
    sos = tokenizer.token_to_id("[SOS]")
    prompt_ids = torch.tensor([[sos]], device=device)

    if video_or_frame.dim() == 3:
        # ── Single-frame path (original behaviour) ──────────────────────
        image   = video_or_frame.unsqueeze(0).to(device)   # (1, C, H, W)
        vis_emb = model.vision_encoder(image)               # (1, N_patches, D)
        proj_emb = model.projector(vis_emb)                 # (1, N_patches, D_llm)

    elif video_or_frame.dim() == 4:
        # ── Full-video path: temporally-aggregated features ──────────────
        # video_or_frame: (C, T, H, W)
        C, T, H, W = video_or_frame.shape
        # Rearrange to (T, C, H, W) and run ImageViT on each frame
        frames = video_or_frame.permute(1, 0, 2, 3).to(device)  # (T, C, H, W)
        # Process in one batched forward pass through the vision encoder
        vis_emb = model.vision_encoder(frames)              # (T, N_patches, D)
        # Temporal mean-pool → collapse time dimension
        vis_emb = vis_emb.mean(dim=0, keepdim=True)        # (1, N_patches, D)
        proj_emb = model.projector(vis_emb)                 # (1, N_patches, D_llm)
    else:
        raise ValueError(f"Expected 3D (C,H,W) or 4D (C,T,H,W) tensor, got {video_or_frame.shape}")

    # ── Autoregressive generation using pre-computed vision embeddings ──
    gen_ids = model.llm.generate(
        input_ids    = prompt_ids,
        vision_embeds= proj_emb,
        max_new_tokens=max_new_tokens,
        temperature  = temperature
    )
    generated_text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)
    return generated_text


# ── Pretty printing ────────────────────────────────────────────────────────────
def print_scores(label: str, scores: dict, true_re: float):
    pred_re_str   = f"{scores['pred_re']:.1f}" if scores["pred_re"] is not None else "not found"
    flow_str      = scores["pred_flow"] or "not found"
    flow_ok       = "[OK]" if scores["flow_match"] else "[X]"

    print(f"\n  {'─'*60}")
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"  Predicted Re:   {pred_re_str}  (ground truth: {true_re:.1f})")
    if scores["re_score"] is not None:
        print(f"  Re accuracy:    {scores['re_score']*100:.1f}%")
    print(f"  Flow type:      {flow_str}  {flow_ok}")
    print(f"  Numeric score:  {scores['numeric_avg']*100:.1f}%")
    print(f"  Physical score: {scores['physical_score']*100:.1f}%")
    if scores["nlp_score"] is not None:
        print(f"  BERTScore F1:   {scores['bert_f1']*100:.1f}%")
        print(f"  ROUGE-L:        {scores['rouge_l']*100:.1f}%")
        print(f"  NLP score:      {scores['nlp_score']*100:.1f}%")
        print(f"  >> PLCS:        {scores['plcs']*100:.1f}%")
    else:
        print(f"  >> Physical PLCS:{scores['plcs']*100:.1f}%  (NLP metrics unavailable)")


def wrap(text, width=60, indent="  "):
    return "\n".join(textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent))


# ── Visualisation ──────────────────────────────────────────────────────────────
def save_comparison_figure(frame_np, true_text, mllm_text, img_text,
                            mllm_scores, img_scores, idx, save_path):
    """Save a PNG showing the frame + both outputs side by side."""
    fig = plt.figure(figsize=(16, 8), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── Frame panel ──
    ax_img = fig.add_subplot(gs[:, 0])
    # frame_np is (C, H, W); C=3 → display as RGB after normalisation
    frame_show = frame_np.transpose(1, 2, 0)  # (H, W, C)
    # Normalise to [0,1] for display
    fmin, fmax = frame_show.min(), frame_show.max()
    frame_show = (frame_show - fmin) / (fmax - fmin + 1e-8)
    ax_img.imshow(frame_show)
    ax_img.set_title(f"Dataset Sample #{idx}\n(displayed frame)", color="white", fontsize=11)
    ax_img.axis("off")

    text_style = dict(transform=fig.transFigure, fontsize=8.5, va="top",
                      color="white", wrap=True)

    # ── Ground truth panel ──
    ax_gt = fig.add_subplot(gs[0, 1:])
    ax_gt.axis("off")
    ax_gt.set_facecolor("#0f3460")
    ax_gt.text(0.01, 0.95, "Ground Truth Caption", transform=ax_gt.transAxes,
               fontsize=10, color="#e2e2e2", fontweight="bold")
    ax_gt.text(0.01, 0.78, "\n".join(textwrap.wrap(true_text, 80)), transform=ax_gt.transAxes,
               fontsize=8.5, color="white", va="top")

    # ── PhysicsMLLM panel ──
    ax_m = fig.add_subplot(gs[1, 1])
    ax_m.axis("off")
    ax_m.set_facecolor("#16213e")
    header = (f"PhysicsMLLM  |  Physical: {mllm_scores['physical_score']*100:.1f}%"
              + (f"  PLCS: {mllm_scores['plcs']*100:.1f}%" if mllm_scores['nlp_score'] else ""))
    ax_m.text(0.02, 0.97, header, transform=ax_m.transAxes,
              fontsize=9, color="#e94560", fontweight="bold", va="top")
    ax_m.text(0.02, 0.80, "\n".join(textwrap.wrap(mllm_text or "(empty)", 42)), transform=ax_m.transAxes,
              fontsize=8, color="white", va="top")

    pred_re_m = f"{mllm_scores['pred_re']:.1f}" if mllm_scores['pred_re'] is not None else "N/A"
    ax_m.text(0.02, 0.12, f"Re predicted: {pred_re_m}", transform=ax_m.transAxes,
              fontsize=8.5, color="#ffd700", va="top")

    # ── ImageMLLM panel ──
    ax_i = fig.add_subplot(gs[1, 2])
    ax_i.axis("off")
    ax_i.set_facecolor("#16213e")
    header = (f"ImageMLLM  |  Physical: {img_scores['physical_score']*100:.1f}%"
              + (f"  PLCS: {img_scores['plcs']*100:.1f}%" if img_scores['nlp_score'] else ""))
    ax_i.text(0.02, 0.97, header, transform=ax_i.transAxes,
              fontsize=9, color="#4cc9f0", fontweight="bold", va="top")
    ax_i.text(0.02, 0.80, "\n".join(textwrap.wrap(img_text or "(empty)", 42)), transform=ax_i.transAxes,
              fontsize=8, color="white", va="top")

    pred_re_i = f"{img_scores['pred_re']:.1f}" if img_scores['pred_re'] is not None else "N/A"
    ax_i.text(0.02, 0.12, f"Re predicted: {pred_re_i}", transform=ax_i.transAxes,
              fontsize=8.5, color="#ffd700", va="top")

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [FIGURE] Comparison figure saved -> {save_path}")


# ── Per-sample evaluation ──────────────────────────────────────────────────────
def evaluate_sample(idx, mllm_model, img_model, tokenizer,
                    mllm_dataset, device, args):
    print(f"\n{SEPARATOR}")
    print(f"  SAMPLE INDEX: {idx}")
    print(SEPARATOR)

    # ── Load raw ground truth directly from HDF5 so both models get the same data ──
    with h5py.File(DATA_PATH, "r") as f:
        keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)
        key  = keys[idx % len(keys)]
        grp  = f[key]

        grid     = grp["grid"][:]                  # (T, C, H, W)  or (T, H, W, C)
        true_re  = float(grp["reynolds_number"][()])
        prompt   = grp["prompt"][()]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

    # ── Ground-truth label (using mllm dataset for consistent generation) ──
    sample_mllm = mllm_dataset[idx]
    true_text   = sample_mllm["text"]

    # Flow regime
    if true_re < 47:
        flow_regime = "laminar"
    elif true_re < 200:
        flow_regime = "unsteady laminar"
    else:
        flow_regime = "turbulent transition"

    print(f"\n  Ground-truth Re:    {true_re:.1f}")
    print(f"  Flow regime:        {flow_regime}")
    print(f"  Original prompt:    {prompt[:120]}{'...' if len(prompt)>120 else ''}")
    print(f"\n  Ground-truth caption:")
    print(wrap(true_text, width=68))

    # ── Run PhysicsMLLM (video input) ──
    video_tensor = sample_mllm["image"]            # (C, T, H, W)
    print(f"\n  >> Running PhysicsMLLM  (video: {list(video_tensor.shape)}) ...")
    mllm_text, mllm_re_pred = run_physics_mllm(mllm_model, tokenizer, video_tensor,
                                               device, args.max_tokens, args.temperature)
    print(f"    Generated: {mllm_text[:120]}{'...' if len(mllm_text)>120 else ''}")
    print(f"    Regression head Re: {mllm_re_pred:.2f}")

    # ── Run ImageMLLM (full video OR single frame, depending on --single_frame flag) ──
    # Pass the entire video so ImageViT can extract per-frame features
    # and mean-pool them across time for richer temporal context.
    if args.single_frame:
        T = video_tensor.shape[1]
        frame_idx = T // 2
        img_input = video_tensor[:, frame_idx, :, :]   # (C, H, W)
        mode_str  = f"single frame {frame_idx}/{T-1}: {list(img_input.shape)}"
    else:
        img_input = video_tensor                        # (C, T, H, W)
        mode_str  = f"full video (T-pooled): {list(img_input.shape)}"

    print(f"\n  >> Running ImageMLLM    ({mode_str}) ...")
    img_text = run_image_mllm(img_model, tokenizer, img_input,
                              device, args.max_tokens, args.temperature)
    print(f"    Generated: {img_text[:120]}{'...' if len(img_text)>120 else ''}")


    # ── Score both ──
    mllm_scores = score_output(true_text, mllm_text,  true_re)
    img_scores  = score_output(true_text, img_text,   true_re)

    # Override Re score for PhysicsMLLM using regression head value directly
    mllm_scores["pred_re"] = mllm_re_pred          # use the explicit head output
    re_err = abs(true_re - mllm_re_pred) / (abs(true_re) + 1e-8)
    mllm_scores["re_score"]    = max(0.0, 1.0 - re_err)
    mllm_scores["numeric_avg"] = mllm_scores["re_score"]   # re is principal numeric for mllm
    mllm_scores["physical_score"] = 0.8 * mllm_scores["numeric_avg"] + 0.2 * float(mllm_scores["flow_match"])
    mllm_scores["plcs"] = (
        (0.7 * mllm_scores["physical_score"] + 0.3 * mllm_scores["nlp_score"])
        if mllm_scores["nlp_score"] is not None
        else mllm_scores["physical_score"]
    )

    print_scores("PhysicsMLLM  [Video ViT + Grassmann + Regression Head]", mllm_scores, true_re)
    print_scores("ImageMLLM    [Image ViT + MLP + Numeric-weighted Loss]",  img_scores,  true_re)

    # ── Verdict ──
    mllm_plcs = mllm_scores["plcs"]
    img_plcs  = img_scores["plcs"]
    diff      = abs(mllm_plcs - img_plcs)
    winner    = "PhysicsMLLM" if mllm_plcs > img_plcs else "ImageMLLM" if img_plcs > mllm_plcs else "TIE"

    print(f"\n  {'─'*60}")
    print(f"  VERDICT  ──  PLCS: PhysicsMLLM {mllm_plcs*100:.1f}%  vs  ImageMLLM {img_plcs*100:.1f}%")
    if winner == "TIE":
        print(f"  [TIE]  (difference < 0.01%)")
    else:
        margin = "marginally" if diff < 0.05 else "clearly"
        print(f"  [WINNER] {winner} wins {margin} by {diff*100:.1f} points")

    # ── Figure ──
    if not args.no_plot:
        frame_np = frame_tensor.cpu().numpy()  # (C, H, W)
        save_path = os.path.join(args.out_dir, f"eval_sample_{idx:04d}.png")
        save_comparison_figure(frame_np, true_text, mllm_text, img_text,
                               mllm_scores, img_scores, idx, save_path)

    return {
        "idx":        idx,
        "true_re":    true_re,
        "mllm_plcs":  mllm_plcs,
        "img_plcs":   img_plcs,
        "winner":     winner,
        "mllm_text":  mllm_text,
        "img_text":   img_text,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{SEPARATOR}")
    print(f"  MLLM Model Comparison Evaluator")
    print(f"  Device: {device}")
    print(SEPARATOR)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load models ──
    print("\n  Loading models...")
    mllm_model, tokenizer = load_physics_mllm(TOKENIZER_PATH, args.mllm_ckpt, device)
    img_model,  _         = load_image_mllm(TOKENIZER_PATH, args.img_ckpt, device)

    # ── Load datasets ──
    print("\n  Loading datasets...")
    mllm_dataset = PhysicsMLLMDataset(DATA_PATH, TOKENIZER_PATH, STAT_PATH, num_frames=24)
    N = len(mllm_dataset)
    print(f"  Dataset size: {N} samples (ImageMLLM will receive a mid-frame slice from each video)")

    # ── Choose indices ──
    if args.idx is not None:
        indices = [args.idx]
    else:
        random.seed(args.seed)
        indices = random.sample(range(N), min(args.n_samples, N))
        indices.sort()
        print(f"  Randomly selected indices (seed={args.seed}): {indices}")

    # ── Evaluate ──
    results = []
    for idx in indices:
        r = evaluate_sample(idx, mllm_model, img_model, tokenizer,
                            mllm_dataset, device, args)
        results.append(r)

    # ── Summary table ──
    if len(results) > 1:
        print(f"\n{SEPARATOR}")
        print(f"  SUMMARY ACROSS {len(results)} SAMPLES")
        print(SEPARATOR)
        print(f"  {'Idx':>5}  {'True Re':>8}  {'PhysicsMLLM':>12}  {'ImageMLLM':>10}  Winner")
        print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*12}")
        for r in results:
            print(f"  {r['idx']:>5}  {r['true_re']:>8.1f}  {r['mllm_plcs']*100:>11.1f}%  "
                  f"{r['img_plcs']*100:>9.1f}%  {r['winner']}")

        mllm_avg = sum(r["mllm_plcs"] for r in results) / len(results)
        img_avg  = sum(r["img_plcs"]  for r in results) / len(results)
        overall_winner = "PhysicsMLLM" if mllm_avg > img_avg else "ImageMLLM" if img_avg > mllm_avg else "TIE"

        print(f"\n  Average PLCS -- PhysicsMLLM: {mllm_avg*100:.1f}%  |  ImageMLLM: {img_avg*100:.1f}%")
        print(f"  [WINNER] Overall winner: {overall_winner}")

    print(f"\n  Done.  Output figures saved to: {os.path.abspath(args.out_dir)}\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PhysicsMLLM vs ImageMLLM")

    parser.add_argument("--idx",         type=int,   default=None,
                        help="Evaluate a specific dataset index (default: random)")
    parser.add_argument("--n_samples",   type=int,   default=1,
                        help="Number of random samples to evaluate (default: 1)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--max_tokens",  type=int,   default=80,
                        help="Max new tokens to generate (default: 80)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--mllm_ckpt",  type=str,   default=MLLM_CKPT,
                        help="Path to PhysicsMLLM checkpoint (.pth)")
    parser.add_argument("--img_ckpt",   type=str,   default=IMG_CKPT,
                        help="Path to ImageMLLM checkpoint (.pth)")
    parser.add_argument("--out_dir",    type=str,   default="eval_output",
                        help="Directory to save comparison figures (default: eval_output)")
    parser.add_argument("--no_plot",     action="store_true",
                        help="Skip saving comparison figures")
    parser.add_argument("--single_frame", action="store_true",
                        help="Feed ImageMLLM a single mid-sequence frame only "
                             "(ablation; default: full video with temporal mean-pool)")

    args = parser.parse_args()
    main(args)
