MECHANISM: Endpoint-weighted monotone local-context taper

HYPOTHESIS: An aligned 144/136/136/136/136/128 local-window schedule will lower val_bpb below 0.983152 by retaining the beneficial early-context bias while avoiding repeated extreme windows and preserving the 136-token average.

INTENDED_EDIT: Replace the current two-level 128/144 allocation with a compute-matched six-layer monotone taper using 144-token endpoints and 136-token middle windows.

EVIDENCE: The aligned 144/144/136/136/128/128 taper achieved the best val_bpb, 0.983152, outperforming the 144/128 split at 0.983275, while the stronger 152/120 bias regressed to 0.983933; this motivates a gentler aligned taper.

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
            17 * long_window // 256,
            17 * long_window // 256,
            17 * long_window // 256,
            17 * long_window // 256,
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
WINDOW_PATTERN = "SSSL" # 144/136/136/136/136/128 local taper; full context at layers 4 and 8
>>>>>>> REPLACE