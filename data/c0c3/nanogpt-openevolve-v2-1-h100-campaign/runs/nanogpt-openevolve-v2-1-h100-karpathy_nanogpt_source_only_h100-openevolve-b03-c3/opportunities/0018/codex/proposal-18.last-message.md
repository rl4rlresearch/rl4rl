MECHANISM: Long-context-selective attention-head gating

HYPOTHESIS: Restricting neutral per-head output gates to the two full-context layers will lower val_bpb below 0.993637 while restoring training volume toward 490M tokens.

INTENDED_EDIT: Remove the unsuccessful MLP gates and apply the proven first-32-channel attention-output gate only on layers whose effective attention window is full context.

EVIDENCE: Per-head attention-output gating achieved the best val_bpb, 0.993637, but reduced volume from 497.0M to 482.3M tokens; alternative gate conditioning and added MLP gating regressed, motivating preservation of the proven gate only where contextual selection is most consequential.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
=======
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        pattern = config.window_pattern.upper()
        is_long_layer = pattern[layer_idx % len(pattern)] == "L" or layer_idx == config.n_layer - 1
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_long_layer else None
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
        y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.out_gate is not None:
            out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
            y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.gate_channels = 32
        self.out_gate = nn.Linear(self.gate_channels, 1, bias=False)

    def forward(self, x):
        gate_input = x[..., :self.gate_channels]
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        out_gate = 2 * torch.sigmoid(self.out_gate(gate_input))
        return x * out_gate
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.out_gate.weight)
            torch.nn.init.zeros_(block.mlp.out_gate.weight)
=======
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.out_gate is not None:
                torch.nn.init.zeros_(block.attn.out_gate.weight)
>>>>>>> REPLACE