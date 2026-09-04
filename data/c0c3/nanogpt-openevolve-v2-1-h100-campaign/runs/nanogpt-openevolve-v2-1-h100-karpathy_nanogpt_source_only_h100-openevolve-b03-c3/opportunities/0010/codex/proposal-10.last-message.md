MECHANISM: Query-conditioned attention head gating

HYPOTHESIS: Replacing fixed-amplitude attention outputs with lightweight, token-dependent per-head gates will improve context selection and lower val_bpb below 0.995200 without materially reducing the roughly 493M-token training volume.

INTENDED_EDIT: Add a zero-initialized sigmoid gate to every attention layer that conditionally attenuates or amplifies each retrieved head. The old assumption is that attention weights alone adequately control contextual influence; the new approach separately learns whether each head’s retrieved context should enter the residual stream.

EVIDENCE: Quarter-context attention reduced training volume from 497.0M to 401.1M tokens and regressed val_bpb to 1.009577, so discarding context is unpromising. Meanwhile, softcap refinements have produced only marginal gains around 0.9952. This patch preserves the proven context windows, batching, and softcap while testing a distinct learned context-selection mechanism with negligible parameter and compute cost.

<<<<<<< SEARCH
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
=======
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
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
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
=======
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.out_gate.weight)
        # Rotary embeddings
>>>>>>> REPLACE