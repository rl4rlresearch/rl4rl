MECHANISM: Downstream-donor post-global context concentration

HYPOTHESIS: With full-context layers 3 and 8, moving eight window tokens from layer 6 to layer 5 while retaining the best-verified 128-token layer-4 window will lower val_bpb below 0.982455.

INTENDED_EDIT: Use 144/144/128/152/120/128 local windows, preserving aggregate attention compute while isolating whether layer 5 benefits from additional context without over-pruning layer 4.

EVIDENCE: The 128/144 layer-4/layer-5 allocation achieved 0.982455, while 120/152 scored a close 0.982539; sourcing the extra layer-5 context from layer 6 retains the best layer-4 setting and uses the same compute-matched window multiset as the 120/152 result.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = 9 * long_window // 128
        late_short_window = long_window // 16
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                short_window = early_short_window if layer_idx < config.n_layer // 2 else late_short_window
                window_sizes.append((short_window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_windows = (
            9 * long_window // 128,
            9 * long_window // 128,
            long_window // 16,
            19 * long_window // 256,
            15 * long_window // 256,
            long_window // 16,
        )
        window_sizes = []
        short_idx = 0
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                window_sizes.append((short_windows[short_idx], 0))
                short_idx += 1
        assert short_idx == len(short_windows)
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144-token early and 128-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144 pre-global; 128/152/120/128 post-global; full context at layers 3 and 8
>>>>>>> REPLACE