MECHANISM: Query-dependent attention-head gating

HYPOTHESIS: Applying a neutral-initialized, query-dependent scalar gate to each attention head will reduce val_bpb below 0.985492 by letting tokens selectively amplify or suppress retrieved context without sacrificing the proven local/global attention topology or meaningful throughput.

INTENDED_EDIT: Challenge the assumption that every attention head’s retrieved context should enter the residual stream at uniform strength: add a learned per-token head gate after FlashAttention, initialized so the model is functionally unchanged at startup.

EVIDENCE: Removing global mixing worsened val_bpb to 0.994122 despite processing more tokens, while batch and schedule refinements have produced only marginal gains; this motivates improving how retrieved context is incorporated rather than reducing attention or further tuning the training schedule.

<<<<<<< SEARCH
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
=======
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # Query-dependent scalar gate controls how strongly each retrieved head
        # contributes while preserving the existing attention distribution.
        self.c_gate = nn.Linear(self.n_embd, self.n_head, bias=False)
        self.ve_gate_channels = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        head_gate = 2 * torch.sigmoid(self.c_gate(x))
        y = y * head_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
=======
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # 2 * sigmoid(0) = 1, so gating is exactly neutral at initialization.
            torch.nn.init.zeros_(block.attn.c_gate.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
>>>>>>> REPLACE