MECHANISM: Cross-stage context reallocation

HYPOTHESIS: Transferring eight window tokens from layer 2 to layer 5 will lower val_bpb below 0.982455 by giving additional context to the strongest post-global recipient while preserving the verified 128-token layer-4 window and aggregate attention compute.

INTENDED_EDIT: Use 144/136/128/152/128/128 local windows with full-context attention at layers 3 and 8.

EVIDENCE: The layer-4-to-layer-5 transfer improved val_bpb from 0.982662 to 0.982455, while reducing layer 2 from 144 to 136 in the front-loaded schedule cost only 0.000034; this identifies layer 2 as a promising donor for layer 5.

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
            17 * long_window // 256,
            long_window // 16,
            19 * long_window // 256,
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
WINDOW_PATTERN = "SSSL" # 144-token early and 128-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/136/128/152/128/128 local windows; full context at layers 3 and 8
>>>>>>> REPLACE