MECHANISM: Depth-progressive local-context allocation

HYPOTHESIS: Using 128-token windows in the first three local layers and 144-token windows in the final three local layers will lower val_bpb below 0.983317 by allocating more context to higher-level representations while matching the total local-attention compute of the best homogeneous 136-token design.

INTENDED_EDIT: Replace uniform 128-token local attention with 128-token early windows and 144-token late windows, retaining full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: Uniform 136-token windows achieved the best val_bpb of 0.983317, outperforming both 128 tokens at 0.983766 and 144 tokens at 0.983758; a 3×128 plus 3×144 schedule preserves exactly the same aggregate local-window budget as 6×136 while testing whether context is more valuable at greater depth.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 16
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 128-token local layers with full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 128-token early and 144-token late local layers; full context at layers 4 and 8
>>>>>>> REPLACE