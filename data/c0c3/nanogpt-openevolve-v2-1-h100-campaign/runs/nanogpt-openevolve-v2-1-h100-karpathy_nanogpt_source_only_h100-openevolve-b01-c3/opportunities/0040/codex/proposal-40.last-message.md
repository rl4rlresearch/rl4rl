MECHANISM: Positionless long-range retrieval head

HYPOTHESIS: On the restored 0.982780 SSSL/aspect-compensated baseline, making one of four heads in each full-context layer positionless will retain at least 445M training tokens and reduce val_bpb below 0.982780 by enabling content-addressed long-range retrieval without distance-dependent rotary phase, while the other three heads preserve ordered context.

INTENDED_EDIT: Restore the verified-best SSSL attention pattern and compensated 5×-MLP Muon rate, then omit RoPE from the final head only in the two full-context layers.

EVIDENCE: The compensated SSSL design achieved the best result, 0.982780. Removing its intermediate global layer regressed to 0.983844, while adding another global layer regressed to 0.985077, suggesting the informative alternative is improving how the proven two global layers retrieve context rather than changing their count.

<<<<<<< SEARCH
    window_pattern: str = "SSSS"
=======
    window_pattern: str = "SSSL"
>>>>>>> REPLACE

<<<<<<< SEARCH
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
=======
def apply_rotary_emb(x, cos, sin, positionless_head=False):
    assert x.ndim == 4
    if positionless_head:
        assert x.size(2) > 1
        x_rope = x[..., :-1, :]
    else:
        x_rope = x
    d = x_rope.shape[3] // 2
    x1, x2 = x_rope[..., :d], x_rope[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    y = torch.cat([y1, y2], 3)
    if positionless_head:
        y = torch.cat([y, x[..., -1:, :]], 2)
    return y
>>>>>>> REPLACE

<<<<<<< SEARCH
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
=======
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        pattern = config.window_pattern.upper()
        self.use_nope_head = pattern[layer_idx % len(pattern)] == "L" or layer_idx == config.n_layer - 1
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
=======
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin, self.use_nope_head)
        k = apply_rotary_emb(k, cos, sin, self.use_nope_head)
        q, k = norm(q), norm(k)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # Preserve the effective MLP update magnitude tuned at 4.375x width.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # final layer is still forced to full context
=======
WINDOW_PATTERN = "SSSL" # two full-context layers; each reserves one NoPE retrieval head
>>>>>>> REPLACE