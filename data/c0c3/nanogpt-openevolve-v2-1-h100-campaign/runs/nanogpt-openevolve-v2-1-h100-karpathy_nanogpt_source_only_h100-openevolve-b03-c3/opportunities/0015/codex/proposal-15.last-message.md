MECHANISM: Fused query-conditioned attention gating

HYPOTHESIS: Fusing the proven first-32-channel attention-output gate with the existing value gate will lower val_bpb below 0.993637 while retaining at least roughly 482M trained tokens.

INTENDED_EDIT: Add neutral per-head attention-output gates, jointly compute value and output gate logits on value-embedding layers to reduce tiny-projection overhead, and use the strongest gated design’s 11.75 softcap.

EVIDENCE: First-slice attention-output gating achieved the best observed val_bpb, 0.993637 on 482.3M tokens; adding a separate MLP gate reduced volume to 477.1M and regressed to 0.994155, motivating a cheaper implementation restricted to the proven attention mechanism.

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
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
=======
        # Compute value and output controls together on value-embedding layers.
        gate_input = x[..., :self.ve_gate_channels]
        if ve is not None:
            gate_logits = F.linear(
                gate_input,
                torch.cat((self.ve_gate.weight, self.out_gate.weight), dim=0),
            )
            ve_gate_logits = gate_logits[..., :self.n_kv_head]
            out_gate_logits = gate_logits[..., self.n_kv_head:]
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + (2 * torch.sigmoid(ve_gate_logits)).unsqueeze(-1) * ve
        else:
            out_gate_logits = self.out_gate(gate_input)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y * (2 * torch.sigmoid(out_gate_logits)).unsqueeze(-1)
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