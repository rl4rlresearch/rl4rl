MECHANISM: Compute-matched expanded local-context taper

HYPOTHESIS: A 152/144/136/136/128/120 local-window taper will lower val_bpb below 0.983152 by concentrating context gradually toward early layers without exposing three layers each to the unsuccessful 152/120 extremes.

INTENDED_EDIT: Replace the current two-level 152/120 allocation with an 8-token-aligned six-local-layer taper that preserves the 136-token average and full-context layers 4 and 8.

EVIDENCE: The 144/144/136/136/128/128 taper achieved the best val_bpb of 0.983152, while the abrupt 152/152/152/120/120/120 split regressed to 0.983933; a gradual expanded taper tests the same endpoint range without the abrupt depth partition.

<<<<<<< SEARCH
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
=======
        short_windows = (
            19 * long_window // 256,
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
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 152-token early and 120-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 152/144/136/136/128/120 local taper; full context at layers 4 and 8
>>>>>>> REPLACE