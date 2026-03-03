"""
evaluate_all_models.py
======================
Side-by-side evaluation of all three MLLMs on the same randomly chosen
dataset sample:

  1. PhysicsMLLM  (src/mllm)        — Video ViT + Grassmann + Regression Head
  2. ImageMLLM    (src/image_mllm)  — Image ViT + MLP, single-frame, numeric loss
  3. VideoMLLM    (src/video_mllm)  — VideoViT (3D) + MLP, full video, numeric loss

Scoring
-------
  Re accuracy     : 1 - |true - pred| / |true|   (from regression head or regex)
  Physical PLCS   : 0.8 * numeric_avg + 0.2 * flow_match
  NLP score       : 0.5 * BERTScore-F1 + 0.5 * ROUGE-L  (optional, falls back)
  PLCS            : 0.7 * physical + 0.3 * nlp  (or physical-only if NLP unavailable)

Usage
-----
  python evaluate_all_models.py                         # 1 random sample
  python evaluate_all_models.py --n_samples 5           # 5 random samples + summary
  python evaluate_all_models.py --idx 487               # specific sample
  python evaluate_all_models.py --single_frame_img      # ImageMLLM: 1 frame (ablation)
  python evaluate_all_models.py --no_video_mllm         # skip if not yet trained
"""

import os, sys, re, io, random, argparse, textwrap, csv, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tokenizers import Tokenizer

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.mllm.model          import PhysicsMLLM
from src.mllm.dataset        import PhysicsMLLMDataset
from src.image_mllm.model_image import ImageMLLM
from src.video_mllm.model_video import VideoMLLM

# ── Default paths ──────────────────────────────────────────────────────────────
DATA_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5"
TOKENIZER_PATH = "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json"
STAT_PATH      = "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl"
CKPT_PHYSICS   = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/mllm_last.pth"
CKPT_IMAGE     = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/image_mllm/best_model.pth"
CKPT_VIDEO     = "d:/Semester 6/Natural Language Processing/Project 3/checkpoints/video_mllm/best_model.pth"

SEP  = "=" * 72
LINE = "-" * 60


# ── Physics value extraction ───────────────────────────────────────────────────
def extract_physics(text: str) -> dict:
    def _f(m, g=1):
        try:
            return float(m.group(g).strip(".,")) if m else None
        except (ValueError, AttributeError):
            return None

    m_re   = re.search(r"(?:Reynolds number\s*is|Reynolds number\s*of|Re\s*=|Re=)\s*([\d\.]+)", text)
    m_rad  = re.search(r"radius(?:\s*of|\s*:)?\s*([\d\.]+)", text)
    m_vel  = re.search(r"velocity(?:\s*of|\s*is|\s*at the inlet is|\s*=)?\s*([\d\.]+)", text)
    m_pos  = re.search(r"(?:position:|located at \(|at\s*\(?|X=)\s*([\d\.]+)(?:,\s*Y=|,\s*)([\d\.]+)", text)
    m_flow = re.search(r"(laminar|unsteady laminar|turbulent transition|turbulent)", text, re.IGNORECASE)

    return {
        "re":     _f(m_re),
        "radius": _f(m_rad),
        "vel":    _f(m_vel),
        "px":     _f(m_pos, 1),
        "py":     _f(m_pos, 2),
        "flow":   m_flow.group(1).lower() if m_flow else None,
    }


def rel_accuracy(true_val, pred_val):
    """0-1 score: 1 = perfect, 0 = missing or very wrong."""
    if true_val is None or pred_val is None:
        return None
    return max(0.0, 1.0 - abs(true_val - pred_val) / (abs(true_val) + 1e-8))


def score_output(true_text: str, pred_text: str, true_re: float,
                 re_from_head: float | None = None) -> dict:
    """
    Compute PLCS breakdown.
    re_from_head : if provided, overrides the regex-extracted Re for the
                   Re accuracy metric (used for PhysicsMLLM's regression head).
    """
    true_phys = extract_physics(true_text)
    pred_phys = extract_physics(pred_text)

    # Re score — use direct head value if available, else regex
    if re_from_head is not None:
        re_score = max(0.0, 1.0 - abs(true_re - re_from_head) / (abs(true_re) + 1e-8))
        pred_re_display = re_from_head
    else:
        re_score = rel_accuracy(true_re, pred_phys["re"])
        pred_re_display = pred_phys["re"]

    rad_score  = rel_accuracy(true_phys["radius"], pred_phys["radius"])
    vel_score  = rel_accuracy(true_phys["vel"],    pred_phys["vel"])
    flow_match = (true_phys["flow"] is not None and pred_phys["flow"] is not None
                  and true_phys["flow"] == pred_phys["flow"])

    numeric_scores = [s for s in [re_score, rad_score, vel_score] if s is not None]
    numeric_avg    = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    physical_score = 0.8 * numeric_avg + 0.2 * float(flow_match)

    # Optional NLP scores
    nlp_score = bert_f1 = rouge_l = None
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
        pass

    plcs = (0.7 * physical_score + 0.3 * nlp_score) if nlp_score is not None else physical_score

    return {
        "plcs":           plcs,
        "physical_score": physical_score,
        "numeric_avg":    numeric_avg,
        "re_score":       re_score,
        "rad_score":      rad_score,
        "vel_score":      vel_score,
        "flow_match":     flow_match,
        "nlp_score":      nlp_score,
        "bert_f1":        bert_f1,
        "rouge_l":        rouge_l,
        "pred_re":        pred_re_display,
        "pred_flow":      pred_phys["flow"],
    }


# ── Model loaders ──────────────────────────────────────────────────────────────
def load_physics(tokenizer_path, ckpt, device):
    tok   = Tokenizer.from_file(tokenizer_path)
    model = PhysicsMLLM(vocab_size=tok.get_vocab_size(), vision_dim=256, llm_dim=256).to(device)
    _load_ckpt(model, ckpt, "PhysicsMLLM")
    model.eval()
    return model, tok


def load_image(tokenizer_path, ckpt, device):
    tok   = Tokenizer.from_file(tokenizer_path)
    model = ImageMLLM(vocab_size=tok.get_vocab_size(), vision_dim=512, llm_dim=512,
                      img_size=64, patch_size=16).to(device)
    _load_ckpt(model, ckpt, "ImageMLLM")
    model.eval()
    return model, tok


def load_video(tokenizer_path, ckpt, device):
    tok   = Tokenizer.from_file(tokenizer_path)
    model = VideoMLLM(vocab_size=tok.get_vocab_size(), vision_dim=512, llm_dim=512,
                      num_frames=24, img_size=64, patch_size=16).to(device)
    _load_ckpt(model, ckpt, "VideoMLLM")
    model.eval()
    return model, tok


def _load_ckpt(model, ckpt, name):
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=next(model.parameters()).device))
        print(f"  [{name:12s}] Loaded: {ckpt}")
    else:
        print(f"  [{name:12s}] [WARNING] Not found: {ckpt}  (random weights)")


# ── Inference helpers ──────────────────────────────────────────────────────────
@torch.no_grad()
def infer_physics(model, tokenizer, video_tensor, device, max_tokens, temperature):
    """(C,T,H,W) -> (text, re_pred_float)"""
    video = video_tensor.unsqueeze(0).to(device)
    vis   = model.vision_encoder(video)
    proj  = model.projector(vis)
    re_p  = model.regression_head(proj.mean(dim=1)).item()
    text  = model.generate(video, tokenizer, max_new_tokens=max_tokens, temperature=temperature)
    return text, re_p


@torch.no_grad()
def infer_image(model, tokenizer, video_tensor, device, max_tokens, temperature, single_frame):
    """(C,T,H,W) -> text   [mean-pool across frames if not single_frame]"""
    sos       = tokenizer.token_to_id("[SOS]")
    prompt_ids = torch.tensor([[sos]], device=device)

    if single_frame:
        T     = video_tensor.shape[1]
        img   = video_tensor[:, T // 2, :, :].unsqueeze(0).to(device)  # (1,C,H,W)
        vis   = model.vision_encoder(img)
        proj  = model.projector(vis)
    else:
        frames = video_tensor.permute(1, 0, 2, 3).to(device)   # (T,C,H,W)
        vis    = model.vision_encoder(frames).mean(dim=0, keepdim=True)  # (1,N,D)
        proj   = model.projector(vis)

    gen  = model.llm.generate(prompt_ids, vision_embeds=proj,
                               max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)


@torch.no_grad()
def infer_video(model, tokenizer, video_tensor, device, max_tokens, temperature):
    """(C,T,H,W) -> text"""
    sos       = tokenizer.token_to_id("[SOS]")
    prompt_ids = torch.tensor([[sos]], device=device)
    video     = video_tensor.unsqueeze(0).to(device)   # (1,C,T,H,W)
    gen_ids   = model.generate(video, prompt_ids,
                                max_new_tokens=max_tokens, temperature=temperature)
    return tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)


# ── Pretty print ──────────────────────────────────────────────────────────────
def print_model_result(label: str, text: str, scores: dict, true_re: float):
    pred_re = f"{scores['pred_re']:.1f}" if scores["pred_re"] is not None else "not detected"
    flow    = scores["pred_flow"] or "not detected"
    flow_ok = "[OK]" if scores["flow_match"] else "[X]"

    print(f"\n  {LINE}")
    print(f"  {label}")
    print(f"  {LINE}")
    print(f"  Generated  : {text[:110]}{'...' if len(text) > 110 else ''}")
    print(f"  Predicted Re      : {pred_re}  (ground truth: {true_re:.1f})")
    if scores["re_score"] is not None:
        print(f"  Re accuracy       : {scores['re_score'] * 100:.1f}%")
    print(f"  Flow type         : {flow}  {flow_ok}")
    print(f"  Numeric score     : {scores['numeric_avg'] * 100:.1f}%")
    print(f"  Physical score    : {scores['physical_score'] * 100:.1f}%")
    if scores["nlp_score"] is not None:
        print(f"  BERTScore F1      : {scores['bert_f1'] * 100:.1f}%")
        print(f"  ROUGE-L           : {scores['rouge_l'] * 100:.1f}%")
        print(f"  NLP score         : {scores['nlp_score'] * 100:.1f}%")
    print(f"  >> PLCS           : {scores['plcs'] * 100:.1f}%"
          + ("" if scores["nlp_score"] is not None else "  (physics-only, NLP unavailable)"))


def wrap(text, width=68, indent="  "):
    return "\n".join(textwrap.wrap(text, width=width,
                                   initial_indent=indent, subsequent_indent=indent))


# ── Comparison figure ──────────────────────────────────────────────────────────
def save_figure(frame_np, true_text, results, idx, save_path):
    """
    results = list of (label_str, text_str, scores_dict, color_hex)
    """
    n_models = len(results)
    fig = plt.figure(figsize=(6 + 5 * n_models, 9), facecolor="#13131f")
    gs  = gridspec.GridSpec(3, 1 + n_models, figure=fig,
                             hspace=0.55, wspace=0.35,
                             height_ratios=[0.15, 0.45, 0.40])

    # ── Frame ──
    ax_img = fig.add_subplot(gs[:, 0])
    show   = frame_np.transpose(1, 2, 0)
    lo, hi = show.min(), show.max()
    show   = (show - lo) / (hi - lo + 1e-8)
    ax_img.imshow(show)
    ax_img.set_title(f"Sample #{idx}", color="white", fontsize=11, pad=6)
    ax_img.axis("off")

    # ── Ground truth row ──
    ax_gt = fig.add_subplot(gs[0, 1:])
    ax_gt.axis("off")
    ax_gt.set_facecolor("#1e2040")
    ax_gt.text(0.01, 0.88, "Ground Truth", transform=ax_gt.transAxes,
               fontsize=10, color="#e2e2e2", fontweight="bold", va="top")
    ax_gt.text(0.01, 0.52, "\n".join(textwrap.wrap(true_text, 100)),
               transform=ax_gt.transAxes, fontsize=7.5, color="white", va="top")

    # ── Per-model panels ──
    for col, (label, text, scores, color) in enumerate(results):
        plcs_str = f"{scores['plcs'] * 100:.1f}%"
        pred_re  = (f"{scores['pred_re']:.1f}" if scores['pred_re'] is not None else "N/A")

        # Generated text panel
        ax_t = fig.add_subplot(gs[1, col + 1])
        ax_t.axis("off")
        ax_t.set_facecolor("#16213e")
        header = f"{label}  |  PLCS: {plcs_str}"
        ax_t.text(0.02, 0.97, header, transform=ax_t.transAxes,
                  fontsize=9, color=color, fontweight="bold", va="top")
        ax_t.text(0.02, 0.78, "\n".join(textwrap.wrap(text or "(empty)", 44)),
                  transform=ax_t.transAxes, fontsize=8, color="white", va="top")
        ax_t.text(0.02, 0.08, f"Re predicted: {pred_re}",
                  transform=ax_t.transAxes, fontsize=8.5, color="#ffd700", va="top")

        # Score bar panel
        ax_s = fig.add_subplot(gs[2, col + 1])
        ax_s.set_facecolor("#16213e")
        metrics = ["Re acc", "Numeric", "Physical", "PLCS"]
        values  = [
            scores["re_score"] or 0.0,
            scores["numeric_avg"],
            scores["physical_score"],
            scores["plcs"],
        ]
        bars = ax_s.barh(metrics, [v * 100 for v in values], color=color, alpha=0.8)
        ax_s.set_xlim(0, 100)
        ax_s.set_xlabel("Score (%)", color="white", fontsize=8)
        ax_s.tick_params(colors="white", labelsize=8)
        ax_s.spines[:].set_color("#444")
        ax_s.set_facecolor("#16213e")
        for bar, val in zip(bars, values):
            ax_s.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                      f"{val*100:.0f}%", va="center", ha="left",
                      fontsize=8, color="white")

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [FIGURE] Saved -> {save_path}")


# ── Per-sample evaluation ──────────────────────────────────────────────────────
def evaluate_sample(idx, models, dataset, device, args):
    """
    models: dict with keys "physics", "image", "video" (may be None if disabled)
    """
    print(f"\n{SEP}")
    print(f"  SAMPLE INDEX: {idx}")
    print(SEP)

    # ── Load ground truth from HDF5 ────────────────────────────────────────
    with h5py.File(DATA_PATH, "r") as f:
        keys   = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
        key    = keys[idx % len(keys)]
        grp    = f[key]
        true_re = float(grp["reynolds_number"][()])
        prompt  = grp["prompt"][()]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

    sample_data = dataset[idx]
    true_text   = sample_data["text"]
    video_tensor = sample_data["image"]   # (C, T, H, W)

    flow_regime = (
        "laminar"            if true_re < 47
        else "unsteady laminar" if true_re < 200
        else "turbulent transition"
    )

    print(f"\n  Ground-truth Re : {true_re:.1f}  ({flow_regime})")
    print(f"  Prompt excerpt  : {prompt[:100]}...")
    print(f"\n  Ground-truth caption:")
    print(wrap(true_text))

    results_list = []   # (label, text, scores, color)

    # ── 1. PhysicsMLLM ───────────────────────────────────────────────────────
    if models["physics"] is not None:
        model_p, tok_p = models["physics"]
        print(f"\n  >> PhysicsMLLM (video: C={video_tensor.shape[0]}, T={video_tensor.shape[1]}) ...")
        t0   = time.time()
        text_p, re_p = infer_physics(model_p, tok_p, video_tensor, device,
                                      args.max_tokens, args.temperature)
        print(f"     Done in {time.time()-t0:.1f}s  |  Re={re_p:.1f}")
        sc_p = score_output(true_text, text_p, true_re, re_from_head=re_p)
        print_model_result("PhysicsMLLM  [Video ViT + Grassmann + Regression Head]",
                           text_p, sc_p, true_re)
        results_list.append(("PhysicsMLLM", text_p, sc_p, "#e94560"))
    else:
        sc_p = None

    # ── 2. ImageMLLM ─────────────────────────────────────────────────────────
    if models["image"] is not None:
        model_i, tok_i = models["image"]
        mode = "single-frame" if args.single_frame_img else "T-pooled video"
        print(f"\n  >> ImageMLLM ({mode}) ...")
        t0   = time.time()
        text_i = infer_image(model_i, tok_i, video_tensor, device,
                              args.max_tokens, args.temperature, args.single_frame_img)
        print(f"     Done in {time.time()-t0:.1f}s")
        sc_i = score_output(true_text, text_i, true_re)
        print_model_result("ImageMLLM    [Image ViT + MLP + Numeric-weighted Loss]",
                           text_i, sc_i, true_re)
        results_list.append(("ImageMLLM", text_i, sc_i, "#4cc9f0"))
    else:
        sc_i = None

    # ── 3. VideoMLLM ─────────────────────────────────────────────────────────
    if models["video"] is not None:
        model_v, tok_v = models["video"]
        print(f"\n  >> VideoMLLM (full video: C={video_tensor.shape[0]}, T={video_tensor.shape[1]}) ...")
        t0   = time.time()
        text_v = infer_video(model_v, tok_v, video_tensor, device,
                             args.max_tokens, args.temperature)
        print(f"     Done in {time.time()-t0:.1f}s")
        sc_v = score_output(true_text, text_v, true_re)
        print_model_result("VideoMLLM    [VideoViT 3D + MLP + Numeric-weighted Loss]",
                           text_v, sc_v, true_re)
        results_list.append(("VideoMLLM", text_v, sc_v, "#b5e48c"))
    else:
        sc_v = None

    # ── Verdict ───────────────────────────────────────────────────────────────
    active_scores = {
        label: sc["plcs"]
        for label, _, sc, _ in results_list
    }
    if active_scores:
        winner = max(active_scores, key=active_scores.get)
        print(f"\n  {LINE}")
        plcs_str = "  |  ".join(f"{k}: {v*100:.1f}%" for k, v in active_scores.items())
        print(f"  VERDICT -- PLCS: {plcs_str}")
        margin = sorted(active_scores.values(), reverse=True)
        diff   = margin[0] - margin[1] if len(margin) > 1 else 0.0
        quality = "clearly" if diff > 0.05 else "marginally"
        print(f"  [WINNER] {winner} wins {quality} by {diff*100:.1f} points")

    # ── Figure ────────────────────────────────────────────────────────────────
    if not args.no_plot and results_list:
        frame_np  = video_tensor[:, video_tensor.shape[1] // 2, :, :].cpu().numpy()
        save_path = os.path.join(args.out_dir, f"eval3_sample_{idx:04d}.png")
        save_figure(frame_np, true_text, results_list, idx, save_path)

    return {
        "idx":       idx,
        "true_re":   true_re,
        "physics_plcs": sc_p["plcs"] if sc_p else None,
        "image_plcs":   sc_i["plcs"] if sc_i else None,
        "video_plcs":   sc_v["plcs"] if sc_v else None,
        "winner":    winner if active_scores else "N/A",
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{SEP}")
    print(f"  Tri-Model MLLM Evaluator")
    print(f"  Device : {device}")
    print(SEP)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n  Loading models...")
    models = {
        "physics": load_physics(TOKENIZER_PATH, args.physics_ckpt, device),
        "image":   load_image(TOKENIZER_PATH, args.image_ckpt, device),
        "video":   None if args.no_video_mllm
                   else load_video(TOKENIZER_PATH, args.video_ckpt, device),
    }

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("\n  Loading dataset...")
    dataset = PhysicsMLLMDataset(DATA_PATH, TOKENIZER_PATH, STAT_PATH, num_frames=24)
    N = len(dataset)
    print(f"  Dataset size: {N}")

    # ── Choose indices ────────────────────────────────────────────────────────
    if args.idx is not None:
        indices = [args.idx]
    else:
        random.seed(args.seed)
        indices = sorted(random.sample(range(N), min(args.n_samples, N)))
        print(f"  Random indices (seed={args.seed}): {indices}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_results = []
    for idx in indices:
        r = evaluate_sample(idx, models, dataset, device, args)
        all_results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{SEP}")
        print(f"  SUMMARY ACROSS {len(all_results)} SAMPLES")
        print(SEP)
        header = f"  {'Idx':>5}  {'True Re':>8}  {'PhysicsMLLM':>12}  {'ImageMLLM':>10}  {'VideoMLLM':>10}  Winner"
        print(header)
        print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}")
        for r in all_results:
            p = f"{r['physics_plcs']*100:>11.1f}%" if r["physics_plcs"] is not None else f"{'N/A':>12}"
            i = f"{r['image_plcs']*100:>9.1f}%"   if r["image_plcs"]   is not None else f"{'N/A':>10}"
            v = f"{r['video_plcs']*100:>9.1f}%"   if r["video_plcs"]   is not None else f"{'N/A':>10}"
            print(f"  {r['idx']:>5}  {r['true_re']:>8.1f}  {p}  {i}  {v}  {r['winner']}")

        # Averages
        avgs = {}
        for key in ("physics_plcs", "image_plcs", "video_plcs"):
            vals = [r[key] for r in all_results if r[key] is not None]
            avgs[key] = sum(vals) / len(vals) if vals else None

        print()
        avg_line = "  Avg PLCS --"
        if avgs["physics_plcs"] is not None:
            avg_line += f"  PhysicsMLLM: {avgs['physics_plcs']*100:.1f}%"
        if avgs["image_plcs"] is not None:
            avg_line += f"  |  ImageMLLM: {avgs['image_plcs']*100:.1f}%"
        if avgs["video_plcs"] is not None:
            avg_line += f"  |  VideoMLLM: {avgs['video_plcs']*100:.1f}%"
        print(avg_line)

        best_key = max((k for k, v in avgs.items() if v is not None), key=avgs.get)
        names    = {"physics_plcs": "PhysicsMLLM", "image_plcs": "ImageMLLM", "video_plcs": "VideoMLLM"}
        print(f"  [OVERALL WINNER] {names[best_key]}")

        # Save CSV summary
        csv_path = os.path.join(args.out_dir, "eval3_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "true_re", "physics_plcs",
                                               "image_plcs", "video_plcs", "winner"])
            w.writeheader()
            w.writerows(all_results)
        print(f"  [CSV] Summary saved -> {csv_path}")

    print(f"\n  Done. Figures -> {os.path.abspath(args.out_dir)}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tri-Model MLLM Evaluator")
    parser.add_argument("--idx",            type=int,   default=None)
    parser.add_argument("--n_samples",      type=int,   default=1)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--max_tokens",     type=int,   default=80)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--physics_ckpt",   type=str,   default=CKPT_PHYSICS)
    parser.add_argument("--image_ckpt",     type=str,   default=CKPT_IMAGE)
    parser.add_argument("--video_ckpt",     type=str,   default=CKPT_VIDEO)
    parser.add_argument("--out_dir",        type=str,   default="eval_output")
    parser.add_argument("--no_plot",        action="store_true")
    parser.add_argument("--single_frame_img", action="store_true",
                        help="Feed ImageMLLM a mid-frame only (ablation)")
    parser.add_argument("--no_video_mllm", action="store_true",
                        help="Skip VideoMLLM (use before training is done)")
    args = parser.parse_args()
    main(args)
