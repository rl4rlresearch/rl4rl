MECHANISM: Compute-neutral staged short-context receptive fields

HYPOTHESIS: Reallocating the uniform short-layer attention budget from 1024/1024/1024 to 768/1024/1280 tokens before each full-context layer will lower val_bpb below 0.992110 while retaining at least 470M training tokens.

INTENDED_EDIT: Restore the best four handoff-layer output gates and progressively expand short-context windows toward each full-context transition without changing total attention-window FLOPs.

EVIDENCE: Four handoff gates achieved the best 0.992110 on 477.1M tokens, and their gain over long-only gating shows predecessor layers matter; the severe regression from uniformly quartering context motivates reallocating rather than reducing the total context budget.

<<<<<<< SEARCH
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
=======
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
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
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.out_gate.weight)
=======
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.out_gate is not None:
                torch.nn.init.zeros_(block.attn.out_gate.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        long_window = config.sequence_len
        assert pattern == "SSSL"
        # Preserve the original total short-attention budget while shifting
        # context toward the representation handed to each full-context layer.
        short_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
>>>>>>> REPLACE