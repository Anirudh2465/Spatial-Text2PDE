"""
dataset_video.py
================
Dataset for VideoMLLM training.

Returns
-------
  image       : (C, T, H, W)   — full normalised video tensor
  input_ids   : (Seq,)          — [SOS] + text tokens + [EOS]
  labels      : (Seq,)          — same as input_ids (teacher-forcing)
  numeric_mask: (Seq,)          — 1.0 at positions with numeric/physics tokens

Text labels
-----------
Uses the original per-simulation `prompt` field from the HDF5 file (which
contains exact physics values like radius, velocity, Re, position, and flow
type).  `variate_prompt` shuffles which sentence template is used during
training to improve generalisation — same approach as ImageCylinderDataset.

Split logic: 80% train / 10% val / 10% test (same ratios as image_mllm).
"""

import re
import random
import torch
import h5py
from torch.utils.data import Dataset
from tokenizers import Tokenizer

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data.normalization import Normalizer


class VideoCylinderDataset(Dataset):
    """
    Full-video dataset for VideoMLLM.

    Each sample is one complete simulation (all T frames). This is in contrast
    to ImageCylinderDataset which returns 4 individual frames per simulation
    (len = 4 × num_simulations).
    Here len = num_simulations in the chosen split.
    """

    def __init__(
        self,
        data_path,
        tokenizer_path,
        stat_path     = None,
        num_frames    = 24,
        split         = "train",
        split_ratios  = (0.8, 0.1, 0.1),
    ):
        self.data_path  = data_path
        self.tokenizer  = Tokenizer.from_file(tokenizer_path)
        self.normalizer = Normalizer(stat_path) if stat_path else None
        self.num_frames = num_frames
        self.split      = split

        with h5py.File(data_path, "r") as f:
            all_keys = sorted(list(f.keys()), key=lambda x: int(x) if x.isdigit() else x)

        n_total = len(all_keys)
        n_train = int(n_total * split_ratios[0])
        n_val   = int(n_total * split_ratios[1])

        if split == "train":
            self.keys = all_keys[:n_train]
        elif split == "val":
            self.keys = all_keys[n_train : n_train + n_val]
        else:
            self.keys = all_keys[n_train + n_val :]

        self.sos_token = self.tokenizer.token_to_id("[SOS]")
        self.eos_token = self.tokenizer.token_to_id("[EOS]")

    # ──────────────────────────────────────────────────────────────────────────
    def variate_prompt(self, prompt: str) -> str:
        """Randomly rephrase the prompt during training; return as-is for val/test."""
        if self.split != "train":
            return prompt

        radius  = re.search(r"radius of ([\d\.]+)", prompt)
        pos     = re.search(r"position: ([\d\.]+),\s*([\d\.]+)", prompt)
        vel     = re.search(r"velocity of ([\d\.]+)", prompt)
        reynolds= re.search(r"Reynolds number is (\d+)", prompt)
        flow_m  = re.search(r"The flow is (.*?)\.", prompt)

        if not (radius and pos and vel and reynolds and flow_m):
            return prompt   # can't parse — return original

        rad    = radius.group(1).rstrip(".")
        px     = pos.group(1).rstrip(".")
        py     = pos.group(2).rstrip(".")
        v      = vel.group(1).rstrip(".")
        re_num = reynolds.group(1).rstrip(".")
        flow   = flow_m.group(1)

        templates = [
            f"Fluid passes over a cylinder with a radius of {rad} and position: {px}, {py}. Fluid enters with a velocity of {v}. The Reynolds number is {re_num}. The flow is {flow}.",
            f"A cylinder of radius {rad} is located at ({px}, {py}) in a fluid stream. The inlet velocity is {v}, resulting in a Reynolds number of {re_num}. We observe that the flow is {flow}.",
            f"The flow is {flow}. It has a Reynolds number of {re_num} and an initial velocity of {v}. This is caused by a cylinder (radius: {rad}) positioned at X={px}, Y={py}.",
            f"With a Reynolds number of {re_num}, the flow is {flow}. The fluid velocity at the inlet is {v}, passing around a cylinder at {px}, {py} with radius {rad}.",
            f"An obstacle cylinder (radius {rad}) at coordinates {px}, {py} interacts with a fluid moving at velocity {v}. The flow is {flow} with Re={re_num}.",
            f"Re = {re_num}. Inlet velocity = {v}. Cylinder radius = {rad} at position ({px}, {py}). The resulting flow state is {flow}.",
        ]
        return random.choice(templates)

    # ──────────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.keys)    # one simulation per index

    def __getitem__(self, idx):
        key = self.keys[idx]

        with h5py.File(self.data_path, "r") as f:
            grp  = f[key]
            grid = grp["grid"][:]   # (T_raw, C, H, W)  or (T_raw, H, W, C)

            prompt = grp["prompt"][()]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")

        # ── Video preparation ──────────────────────────────────────────────
        grid_t = torch.tensor(grid).float()   # (T_raw, C, H, W)

        if grid_t.shape[0] > self.num_frames:
            grid_t = grid_t[: self.num_frames]

        # Normalise  (Normalizer expects B, T, C, H, W)
        if self.normalizer:
            grid_t = self.normalizer.normalize(grid_t.unsqueeze(0)).squeeze(0)

        # (T, C, H, W) → (C, T, H, W)  ← model input convention
        image = grid_t.permute(1, 0, 2, 3)

        # ── Text preparation ───────────────────────────────────────────────
        caption    = self.variate_prompt(prompt)
        text_tokens = self.tokenizer.encode(caption).ids
        ids         = [self.sos_token] + text_tokens + [self.eos_token]

        input_tensor = torch.tensor(ids, dtype=torch.long)
        labels       = input_tensor.clone()

        # Numeric mask: 1.0 at token positions that contain digits, '=', or '.'
        numeric_mask = torch.zeros_like(input_tensor, dtype=torch.float)
        for i, tid in enumerate(ids):
            token = self.tokenizer.decode([tid])
            if any(c.isdigit() for c in token) or token in ["=", "."]:
                numeric_mask[i] = 1.0

        return {
            "image":        image,          # (C, T, H, W)
            "input_ids":    input_tensor,   # (Seq,)
            "labels":       labels,         # (Seq,)
            "numeric_mask": numeric_mask,   # (Seq,)
            "text":         caption,        # raw string (for inspection)
        }


if __name__ == "__main__":
    ds = VideoCylinderDataset(
        "d:/Semester 6/Natural Language Processing/Project 3/train_grid_64.h5",
        "d:/Semester 6/Natural Language Processing/Project 3/src/mllm/mllm_tokenizer.json",
        "d:/Semester 6/Natural Language Processing/Project 3/train_normal_stat.pkl",
        num_frames=24,
        split="train",
    )
    print(f"Dataset size: {len(ds)}")
    s = ds[0]
    print("image        :", s["image"].shape)
    print("input_ids    :", s["input_ids"].shape)
    print("numeric_mask :", s["numeric_mask"].sum().item(), "numeric tokens")
    print("caption      :", s["text"])
