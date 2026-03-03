"""
model_video.py
==============
VideoMLLM: Multimodal LLM that processes full video sequences.

Architecture
------------
  VideoViT      (B, C, T, H, W) → (B, 384, 512)   spatio-temporal tokens
  VideoProjector (B, 384, 512)  → (B, 384, 512)   MLP feature bridge
  VideoGPT      [vision_prefix | text_tokens]      causal language decoder
                                                    numeric-token upweighted loss

Key differences from src/image_mllm/model_image.py
  • VideoViT replaces ImageViT — processes all T frames, not just 1.
  • Vision prefix is 384 tokens (T×H'×W') not 16 tokens (H'×W').
  • Re is predicted purely through language generation (no separate head),
    identical to the image_mllm design.
  • Checkpoints saved to checkpoints/video_mllm/, separate from other models.

Key differences from src/mllm/model.py (PhysicsMLLM)
  • No GrassmannProjector / SVD — uses a plain MLP projector.
  • No explicit regression head — Re lives in the generated text.
  • Loss: CE with numeric token upweighting (not CE + MSE).
"""

import torch
import torch.nn as nn
from src.video_mllm.vision_video   import VideoViT
from src.video_mllm.language_video import VideoGPT, VideoLLMConfig


class VideoProjector(nn.Module):
    """
    Plain MLP that bridges the video encoder output dimension to the LLM
    embedding dimension.  Input and output share the same dim=512 in the
    default configuration, but the MLP allows non-linear feature adaptation.
    """
    def __init__(self, input_dim: int, llm_dim: int, mlp_depth: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, llm_dim), nn.GELU()]
        for _ in range(mlp_depth - 1):
            layers += [nn.Linear(llm_dim, llm_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)   # (B, K, llm_dim)


class VideoMLLM(nn.Module):
    """
    Full VideoMLLM model.

    Parameters
    ----------
    vocab_size  : tokenizer vocab size
    vision_dim  : VideoViT embedding dimension
    llm_dim     : LLM embedding dimension
    num_frames  : temporal depth of the video (default 24)
    img_size    : spatial size of each frame (default 64)
    patch_size  : spatial patch size (default 16)
    """
    def __init__(
        self,
        vocab_size  = 5000,
        vision_dim  = 512,
        llm_dim     = 512,
        num_frames  = 24,
        img_size    = 64,
        patch_size  = 16,
    ):
        super().__init__()

        # 1. Vision encoder
        self.vision_encoder = VideoViT(
            img_size   = img_size,
            patch_size = patch_size,
            num_frames = num_frames,
            embed_dim  = vision_dim,
        )

        # 2. Projector
        self.projector = VideoProjector(
            input_dim = vision_dim,
            llm_dim   = llm_dim,
            mlp_depth = 2,
        )

        # 3. Language decoder
        config = VideoLLMConfig(
            vocab_size = vocab_size,
            n_embd     = llm_dim,
            n_head     = 8,
            n_layer    = 8,
            max_len    = 1024,
        )
        self.llm     = VideoGPT(config)
        self.llm_dim = llm_dim

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        video,
        input_ids,
        targets             = None,
        numeric_mask        = None,
        numeric_loss_lambda = 1.0,
    ):
        """
        Args:
            video       : (B, C, T, H, W)
            input_ids   : (B, T_text)
            targets     : (B, T_text)  — -100 for pad
            numeric_mask: (B, T_text)  — 1.0 for numeric tokens
        Returns:
            logits      : (B, K + T_text, vocab_size)
            loss        : scalar or None
        """
        vis_emb  = self.vision_encoder(video)       # (B, 384, vision_dim)
        proj_emb = self.projector(vis_emb)           # (B, 384, llm_dim)

        logits, loss = self.llm(
            input_ids           = input_ids,
            vision_embeds       = proj_emb,
            targets             = targets,
            numeric_mask        = numeric_mask,
            numeric_loss_lambda = numeric_loss_lambda,
        )
        return logits, loss

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, video, input_ids, max_new_tokens=80, temperature=1.0):
        """
        Autoregressive text generation from a video.
        Args:
            video      : (1, C, T, H, W)
            input_ids  : (1, L)  — e.g. [[SOS]]
        Returns:
            generated token id tensor (1, L + new_tokens)
        """
        self.eval()
        vis_emb  = self.vision_encoder(video)
        proj_emb = self.projector(vis_emb)
        return self.llm.generate(
            input_ids     = input_ids,
            vision_embeds = proj_emb,
            max_new_tokens= max_new_tokens,
            temperature   = temperature,
        )


if __name__ == "__main__":
    model = VideoMLLM()
    video = torch.randn(2, 3, 24, 64, 64)
    ids   = torch.randint(0, 5000, (2, 40))
    tgt   = torch.randint(0, 5000, (2, 40))
    mask  = (torch.rand(2, 40) > 0.8).float()
    logits, loss = model(video, ids, targets=tgt, numeric_mask=mask, numeric_loss_lambda=5.0)
    print("Logits :", logits.shape)
    print("Loss   :", loss.item())
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {params/1e6:.2f}M")
