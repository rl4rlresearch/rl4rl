MECHANISM: Terminal-local context pruning

HYPOTHESIS: A 144/144/136/136/128/120 local-window schedule will lower val_bpb below 0.983152 by retaining the best taper’s useful early context while shortening the local layer immediately before final full-context integration to gain throughput.

INTENDED_EDIT: Replace the current early/late split with the best verified explicit taper, except reduce its final local window from 128 to 120 tokens.

EVIDENCE: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, while moving a larger 136-token window to the final local layer regressed to 0.983660; this suggests additional context is least valuable immediately before the final full-context layer.

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
            17 * long_window // 256,
            17 * long_window // 256,
            long_window // 16,
            15 * long_window // 256,
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
WINDOW_PATTERN = "SSSL" # 144/144/136/136/128/120 local taper; full context at layers 4 and 8
>>>>>>> REPLACE