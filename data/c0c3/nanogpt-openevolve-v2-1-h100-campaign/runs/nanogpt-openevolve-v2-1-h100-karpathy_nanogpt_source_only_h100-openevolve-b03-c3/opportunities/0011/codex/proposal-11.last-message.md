MECHANISM: Full-state query-conditioned attention-head gating

HYPOTHESIS: Conditioning the proven per-head attention-output gates on all 512 hidden channels instead of an arbitrary 32-channel slice will lower val_bpb below 0.993637 while retaining roughly 480M trained tokens.

INTENDED_EDIT: Add neutral-initialized per-head attention-output gates using the full normalized token state, and use the 11.75 softcap from the strongest gated reference design.

EVIDENCE: Reference Design 3’s 32-channel attention-output gates improved val_bpb from 0.995200 to 0.993637 while still training 482.3M tokens; this motivates preserving the gate and testing whether complete-state conditioning improves its context-selection signal.

<<<<<<< SEARCH
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
=======
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.n_embd, self.n_head, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 2 * torch.sigmoid(self.out_gate(x))
        y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
=======
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.out_gate.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x).float()
        softcap = 12.0
        logits = softcap * torch.tanh(logits / softcap)
=======
        logits = self.lm_head(x).float()
        softcap = 11.75
        logits = softcap * torch.tanh(logits / softcap)
>>>>>>> REPLACE