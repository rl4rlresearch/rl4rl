MECHANISM: Query-conditioned attention write gates

HYPOTHESIS: Starting from the best verified projection-only Muon compensation, neutral-initialized per-token head gates will retain at least 445M training tokens and reduce `val_bpb` below 0.982763 by letting attention selectively suppress or amplify retrieved information before it enters the residual stream.

INTENDED_EDIT: Replace the assumption that every attention head must always write with a learned query-dependent no-op/amplification gate, while restoring the verified-best contraction-only learning-rate adjustment.

EVIDENCE: Projection-only `sqrt(7/8)` compensation achieved 0.982763. Changing global-layer counts and making a whole head positionless regressed, motivating a mechanism that improves how the proven attention pattern writes context without changing its receptive fields, rotary representation, or number of attention calls.

<<<<<<< SEARCH
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
=======
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
        y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
=======
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.out_gate.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
=======
            # Preserve the best verified compensation only for the 512x2560
            # MLP contractions; expansion and attention remain at the base rate.
            mlp_lr_scale = math.sqrt(7 / 8) if shape[1] == 5 * shape[0] else 1.0
>>>>>>> REPLACE