MECHANISM: Isolated post-global context expansion

HYPOTHESIS: Increasing only the first post-global local window from 136 to 144 tokens will lower val_bpb below 0.982662 by strengthening the value-embedding layer immediately after global mixing without the confounding terminal-window pruning used previously.

INTENDED_EDIT: Restore full-context layers 3 and 8 with the best verified taper, then expand layer 4 alone to produce 144/144/144/136/128/128 local windows while retaining all optimizer settings.

EVIDENCE: The 144/144/136/136/128/128 layer-3/layer-8 design achieved the best val_bpb of 0.982662. Its compute-matched post-global restart scored 0.982945 but simultaneously expanded layer 4 and pruned layer 7; independently, terminal pruning regressed, motivating isolation of the layer-4 expansion.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = 19 * long_window // 256
        late_short_window = 15 * long_window // 256
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
            9 * long_window // 128,
            17 * long_window // 256,
            long_window // 16,
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
WINDOW_PATTERN = "SSSL" # 152-token early and 120-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/144/136/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE