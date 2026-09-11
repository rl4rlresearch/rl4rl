MECHANISM: Steeper compute-neutral staged short-context attention

HYPOTHESIS: Using 512/1024/1536-token short windows before each full-context layer, together with the proven four handoff gates, will lower val_bpb below 0.991835 while retaining at least 480M training tokens.

INTENDED_EDIT: Restore output gating on full-context layers and their immediate predecessors, then redistribute each three-layer short-attention budget from 1024/1024/1024 to 512/1024/1536 tokens.

EVIDENCE: The milder 768/1024/1280 progression with four handoff gates achieved the best result, 0.991835 on 480.2M tokens, improving over uniform windows at 0.992110; a steeper progression tests whether concentrating still more context at the handoff extends that gain without reducing total window budget.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
        is_long_layer = pattern[layer_idx % len(pattern)] == "L" or layer_idx == config.n_layer - 1
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_long_layer else None
=======
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        assert pattern == "SSSL"
        # Keep the original total short-attention budget, but concentrate
        # context in the representation immediately handed to each long layer.
        short_windows = (
            long_window // 4,
            long_window // 2,
            3 * long_window // 4,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE