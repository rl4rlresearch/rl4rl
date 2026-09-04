MECHANISM: Depth-dependent local-attention windows

HYPOTHESIS: Using 320-token windows in the first three local layers and 416-token windows in the later three will reduce val_bpb below 0.983765 by recovering some throughput while preserving the best-performing context size near the output.

INTENDED_EDIT: Make early short-attention layers use 320 tokens while retaining 416 tokens for later short-attention layers and full context for both existing global layers.

EVIDENCE: Uniform 320-token attention processed 509.1M tokens but reached 0.983911, while uniform 416-token attention achieved the best 0.983765 with 506.5M tokens; this hybrid tests whether the 416-token quality benefit is concentrated in later layers.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = 13 * long_window // 64
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        long_window = config.sequence_len
        short_window = 13 * long_window // 64
        early_short_window = 5 * long_window // 32
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                window = early_short_window if layer_idx < config.n_layer // 2 else short_window
                window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
>>>>>>> REPLACE