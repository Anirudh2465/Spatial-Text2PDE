"""
language_video.py
=================
VideoGPT: GPT-style causal language decoder for the video_mllm.

Identical in logic to ImageGPT (language_image.py) but with max_len
extended to 1024 to accommodate the larger vision prefix from VideoViT:
  - ImageViT   → 16 vision tokens  (4×4 patches, 1 frame)
  - VideoViT   → 384 vision tokens (4×4 patches × 24 frames)
So position IDs can reach up to 384 + ~80 text tokens = ~464.
We use max_len=1024 for safe headroom.

Weight-sharing: token_embedding ↔ output head (weight tying).
"""

import torch
import torch.nn as nn


class VideoLLMConfig:
    def __init__(
        self,
        vocab_size  = 5000,
        max_len     = 1024,
        n_embd      = 512,
        n_head      = 8,
        n_layer     = 8,
        dropout     = 0.1,
    ):
        self.vocab_size  = vocab_size
        self.max_len     = max_len
        self.n_embd      = n_embd
        self.n_head      = n_head
        self.n_layer     = n_layer
        self.dropout     = dropout


class VideoGPT(nn.Module):
    def __init__(self, config: VideoLLMConfig):
        super().__init__()
        self.config = config

        self.token_embedding    = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_len,    config.n_embd)
        self.drop               = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = config.n_embd,
            nhead          = config.n_head,
            dim_feedforward= 4 * config.n_embd,
            dropout        = config.dropout,
            activation     = "gelu",
            batch_first    = True,
            norm_first     = True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.ln_f   = nn.LayerNorm(config.n_embd)
        self.head   = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.head.weight
        self.apply(self._init_weights)

    # ──────────────────────────────────────────────────────────────────────────
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        vision_embeds       = None,
        targets             = None,
        numeric_mask        = None,
        numeric_loss_lambda = 1.0,
    ):
        """
        Args:
            input_ids           : (B, T_text)
            vision_embeds       : (B, K, D)   K = number of vision tokens (384 for VideoViT)
            targets             : (B, T_text)  -100 for ignored positions
            numeric_mask        : (B, T_text)  1.0 for numeric/physics tokens
            numeric_loss_lambda : extra weight on numeric-token CE loss

        Returns:
            logits : (B, K + T_text, vocab_size)
            loss   : scalar or None
        """
        b, t_text = input_ids.size()
        device    = input_ids.device

        tok_emb = self.token_embedding(input_ids)   # (B, T_text, D)

        if vision_embeds is not None:
            k   = vision_embeds.size(1)
            x   = torch.cat([vision_embeds, tok_emb], dim=1)   # (B, K+T, D)
            seq_len = x.size(1)

            if seq_len > self.config.max_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_len {self.config.max_len}. "
                    f"Reduce num_frames, patch_size, or text length."
                )

            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            x   = self.drop(x + self.position_embedding(pos))

            # Causal mask; vision tokens attend to each other fully (bi-directional)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
            )
            mask[:k, :k] = 0.0   # vision prefix: full attention among themselves

            x      = self.blocks(x, mask=mask, is_causal=False)
            x      = self.ln_f(x)
            logits = self.head(x)

            if targets is not None:
                # v_k predicts t_1, so slice logits[k-1 : -1]
                text_logits = logits[:, k - 1 : -1, :].contiguous()
                targets     = targets.contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                ce_loss  = loss_fct(
                    text_logits.view(-1, self.config.vocab_size),
                    targets.view(-1)
                ).view(b, t_text)

                if numeric_mask is not None:
                    numeric_loss = ce_loss * numeric_mask
                    ce_loss      = ce_loss + numeric_loss_lambda * numeric_loss

                valid = targets != -100
                loss  = ce_loss[valid].mean() if valid.sum() > 0 else torch.tensor(
                    0.0, device=device, requires_grad=True
                )
            else:
                loss = None

        else:
            # Text-only path
            pos  = torch.arange(t_text, dtype=torch.long, device=device).unsqueeze(0)
            x    = self.drop(tok_emb + self.position_embedding(pos))
            mask = torch.triu(
                torch.ones(t_text, t_text, device=device) * float("-inf"), diagonal=1
            )
            x      = self.blocks(x, mask=mask, is_causal=True)
            x      = self.ln_f(x)
            logits = self.head(x)

            loss = None
            if targets is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()
                loss_fct     = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                ce_loss      = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                ).view(b, t_text - 1)

                if numeric_mask is not None:
                    shift_numeric = numeric_mask[:, 1:].contiguous()
                    numeric_loss  = ce_loss * shift_numeric
                    ce_loss       = ce_loss + numeric_loss_lambda * numeric_loss

                valid = shift_labels != -100
                loss  = ce_loss[valid].mean() if valid.sum() > 0 else torch.tensor(
                    0.0, device=device, requires_grad=True
                )

        return logits, loss

    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, input_ids, vision_embeds=None, max_new_tokens=80, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids, vision_embeds)
            logits     = logits[:, -1, :]
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs    = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        return input_ids
