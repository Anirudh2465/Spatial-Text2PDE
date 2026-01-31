import torch
import torch.nn as nn
import math

class PhysicsGPTConfig:
    def __init__(self, vocab_size, max_len=128, n_embd=256, n_head=4, n_layer=4, dropout=0.1):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

class PhysicsGPT(nn.Module):
    def __init__(self, config: PhysicsGPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, vision_embeds=None, targets=None, numeric_mask=None, numeric_loss_lambda=1.0, physics_end_token_id=5):
        # idx: (b, t_text)
        # vision_embeds: (b, k, n_embd) -- Projected vision tokens
        
        b, t_text = idx.size()
        device = idx.device
        
        # Token Embeddings
        tok_emb = self.token_embedding(idx) # (b, t_text, n_embd)
        
        if vision_embeds is not None:
            # Sequence: [Physics] [Vision] [Text]
            # Requires splitting idx into Physics and Text.
            # We assume idx contains the full Sequence [Physics... </PHYSICS> Text...]
            # But wait, the standard Dataset returns the full sequence.
            # We need to insert Vision tokens in the middle.
            # Find </PHYSICS> (id=5 usually, pass as arg or find)
            
            # This splits the batch effectively? 
            # It's cleaner if we pass physics_ids and text_ids separately, but we have full input_ids.
            # Let's find the split point per sample?
            # Or assume we can just concatenate logic?
            # 
            # Prompt: [Physics Tokens] [Vision Tokens] [Text Tokens]
            # Input `idx` has [Physics Tokens] ... [Text Tokens]
            # We need to splice `vision_embeds` in between.
            
            # Since split point might vary per sample, masking is complex.
            # Limitation: Assume split point is consistent? No, number of digits varies.
            # We can construct the embedding sequence per sample.
            
            full_emb_list = []
            max_seq_len = 0
            
            # We also need to build the custom mask per sample
            # This is expensive to do in loop.
            # optimize: find physics_end per row
            
            # physics_end_token_id = self.config.physics_end_token_id usually
            # let's assume passed in args for now
             
            # Indices of </PHYSICS>
            # (B,)
            split_indices = (idx == physics_end_token_id).nonzero(as_tuple=True)
            # This implies one per row.
            
            # But iterating is slow.
            # Easier way:
            # 1. Embed all text.
            # 2. Slice Physics = tok_emb[:, :split+1]
            # 3. Slice Text = tok_emb[:, split+1:]
            # 4. Cat Physics + Vision + Text
            
            # However, batching variable lengths is hard without padding.
            # Simplified approach:
            # Just Insert Vision at the end of Physics?
            # For Phase 2 Training, we can ensure fixed physics length?
            # "No variation in ordering or syntax".
            # "Re = 230 ;" -> 5 tokens.
            # "Re = 2300 ;" -> 6 tokens.
            # Varies.
            
            # We will use mask generation.
            
            # Actually, standard Transformer supports mask.
            # For each sample, we construct the sequence.
            
            # Let's try to do it in batch if possible.
            # We need to compute the combined embeddings.
            
            combined_embs = []
            masks = []
            
            k = vision_embeds.size(1)
            
            for i in range(b):
                row_idx = idx[i]
                # find split
                split_candidates = (row_idx == physics_end_token_id).nonzero()
                if len(split_candidates) == 0:
                     # Fallback (maybe during generation?), append vision at start?
                     # Or just treat everything as Text.
                     phys_end = 0
                else:
                    phys_end = split_candidates[0].item() + 1 # Include </PHYSICS>
                
                phys_emb = tok_emb[i, :phys_end, :]
                text_emb = tok_emb[i, phys_end:, :]
                vis_emb = vision_embeds[i]
                
                # Concat: [Phys, Vis, Text]
                c_emb = torch.cat([phys_emb, vis_emb, text_emb], dim=0) # (T_total, D)
                combined_embs.append(c_emb)
                
                # Create Attention Mask
                # Size (T_total, T_total)
                # Rules:
                # P (0:phys_end) -> P
                # V (phys_end:phys_end+k) -> P, V
                # T (phys_end+k:) -> P, V, T (Causal)
                
                t_total = c_emb.size(0)
                mask = torch.full((t_total, t_total), float('-inf'), device=device)
                
                # P -> P (Bidirectional)
                mask[:phys_end, :phys_end] = 0
                
                # V -> P, V
                v_start = phys_end
                v_end = v_start + k
                mask[v_start:v_end, :v_end] = 0
                
                # T -> P, V, T (Causal)
                t_start = v_end
                # T can attend to P (0:phys_end) and V (v_start:v_end) and itself up to current position
                # Standard causal mask for T part relative to T part
                # mask[i, j] = 0 if j <= i
                
                # Set P and V visibility for T
                mask[t_start:, :t_start] = 0
                
                # Set T causal visibility
                # i range(t_start, t_total)
                # j range(t_start, t_total)
                # causal
                
                causal_block = torch.triu(torch.ones(t_total - t_start, t_total - t_start, device=device) * float('-inf'), diagonal=1)
                mask[t_start:, t_start:] = causal_block
                
                masks.append(mask)
            
            # Pad combined embeddings to max length in batch
            max_len = max([e.size(0) for e in combined_embs])
            padded_embs = torch.zeros(b, max_len, self.config.n_embd, device=device)
            padded_masks = torch.zeros(b, self.config.n_head, max_len, max_len, device=device) # n_head for multihead
            
            for i in range(b):
                l = combined_embs[i].size(0)
                padded_embs[i, :l, :] = combined_embs[i]
                # Extend mask: pad area is masked
                # Current mask is (l, l). Padded is (max, max).
                # We need to mask all attention TO padding.
                # Initialize full mask with -inf
                m = torch.full((max_len, max_len), float('-inf'), device=device)
                m[:l, :l] = masks[i]
                # Repeat for heads
                padded_masks[i, :, :, :] = m.unsqueeze(0)
                
            x = padded_embs
            # pos_emb: we need to adjust position ids?
            # [P][V][T]
            # Pos 0..P-1
            # Pos P..P+K-1 ? Or V resets? "Learned positional embeddings".
            # Usually absolute.
            pos = torch.arange(0, max_len, device=device).unsqueeze(0)
            x = x + self.position_embedding(pos) # simplified, assumes max_len fits
            x = self.drop(x)
            
            # Transformer
            # Custom mask shape: (B*n_head, T, T) or (B, n_head, T, T)
            # nn.TransformerEncoder expects mask (T, T) or (B*nhead, T, T)
            # We flattened batch and head.
            # mask: (B, H, T, T) -> (B*H, T, T)
            mask = padded_masks.view(b * self.config.n_head, max_len, max_len)
            
            x = self.blocks(x, mask=mask, is_causal=False) # Mask handles causality
            
            # Output Logits
            x = self.ln_f(x)
            logits = self.head(x)
            
            # Calculate Loss using the shifted logic on TEXT part only?
            # Targets need to be modified: [P_labels][V_labels][T_labels]
            # P_labels: -100
            # V_labels: -100
            # T_labels: original labels[phys_end:]
            
            if targets is not None:
                total_loss = 0
                count = 0
                
                # Reconstruct targets for the batch
                new_targets = torch.full((b, max_len), -100, dtype=torch.long, device=device)
                new_numeric_mask = torch.zeros((b, max_len), dtype=torch.float, device=device)
                
                for i in range(b):
                    row_idx = idx[i]
                    split_candidates = (row_idx == physics_end_token_id).nonzero()
                    if len(split_candidates) > 0:
                        phys_end = split_candidates[0].item() + 1
                        # Old target segment for text
                        # original targets: [P... T....]
                        # We want to map T targets to new coordinates
                        # Text starts at phys_end in input.
                        # In new sequence, Text starts at phys_end + k.
                        
                        orig_text_targets = targets[i, phys_end:]
                        orig_numeric_mask = numeric_mask[i, phys_end:] if numeric_mask is not None else None
                        
                        l_text = orig_text_targets.size(0)
                        
                        # Placement
                        start_new = phys_end + k
                        new_targets[i, start_new : start_new + l_text] = orig_text_targets
                        if orig_numeric_mask is not None:
                            new_numeric_mask[i, start_new : start_new + l_text] = orig_numeric_mask
                
                # Shift for autoregressive loss
                # logits[t] predicts targets[t+1]
                # But here targets are aligned with input x? 
                # Usually: input [A, B, C], target [B, C, D]
                # Our new_targets are "Next Token" labels aligned with 'input' indices?
                # No, we just copied from 'targets' which are usually inputs.
                # So we need shifting.
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = new_targets[:, 1:].contiguous()
                shift_numeric = new_numeric_mask[:, 1:].contiguous() if numeric_mask is not None else None
                
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                ce_loss = ce_loss.view(b, max_len-1)
                
                if shift_numeric is not None:
                    numeric_loss = ce_loss * shift_numeric
                    ce_loss = ce_loss + numeric_loss_lambda * numeric_loss
                    
                loss = ce_loss[shift_labels != -100].mean()
            else:
                loss = None

        else:
            # Phase 1 Legacy Path (Standard)
            pos = torch.arange(0, t_text, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
            
            # Embeddings
            tok_emb = self.token_embedding(idx) # (b, t, n_embd)
            pos_emb = self.position_embedding(pos) # (1, t, n_embd)
            x = self.drop(tok_emb + pos_emb)
            mask = torch.triu(torch.ones(t_text, t_text, device=device) * float('-inf'), diagonal=1)
            x = self.blocks(x, mask=mask, is_causal=True)
            x = self.ln_f(x)
            logits = self.head(x) 
            
            loss = None
            if targets is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = targets[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                ce_loss = ce_loss.view(b, t_text-1)
                if numeric_mask is not None:
                    shift_numeric_mask = numeric_mask[:, 1:].contiguous()
                    numeric_loss = ce_loss * shift_numeric_mask
                    active_loss = ce_loss + numeric_loss_lambda * numeric_loss
                    loss = active_loss[shift_labels != -100].mean()
                else:
                    loss = ce_loss[shift_labels != -100].mean()
                
        return logits, loss
                
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, vision_embeds=None, max_new_tokens=30, temperature=1.0, top_k=None, physics_end_token_id=5):
        """
        Conditional generation.
        If vision_embeds is provided, we assume idx contains [B, P...].
        This method is simplified for test time: 
        We rely on the forward pass to integrate vision.
        """
        for _ in range(max_new_tokens):
            # Crop to context window if needed (Not implemented for vision complex case yet)
            
            # Forward
            logits, _ = self.forward(idx, vision_embeds=vision_embeds, physics_end_token_id=physics_end_token_id)
            logits = logits[:, -1, :]
            
            if temperature == 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
