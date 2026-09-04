MECHANISM: Early global-context integration

HYPOTHESIS: Moving the first full-context layer from layer 4 to layer 3 while retaining the best 144/144/136/136/128/128 local-window taper will lower val_bpb below 0.983152 by giving five downstream layers access to globally mixed representations without changing aggregate attention compute.

INTENDED_EDIT: Restore the best verified six-window taper and place full-context attention at layers 3 and 8.

EVIDENCE: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, and its larger early windows indicate that context is especially valuable near the bottom of the network; advancing the first full-context layer tests that implication on a new architectural axis while preserving the exact local-window multiset.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = long_window // 16
        late_short_window = 9 * long_window // 128
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
            17 * long_window // 256,
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
WINDOW_PATTERN = "SSSL" # 128-token early and 144-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSLSSSSL" # 144/144/136/136/128/128 local taper; full context at layers 3 and 8
>>>>>>> REPLACE