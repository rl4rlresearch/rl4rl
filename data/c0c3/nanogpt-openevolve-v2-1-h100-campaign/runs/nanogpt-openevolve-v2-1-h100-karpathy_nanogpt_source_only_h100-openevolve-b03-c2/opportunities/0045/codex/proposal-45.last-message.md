MECHANISM: Front-loaded local context allocation

HYPOTHESIS: Using 512-token local attention before the intermediate full-context layer and 256-token local attention afterward will reduce `val_bpb` below 0.983618 by preserving broader token interactions during early feature construction while using cheaper late refinement.

INTENDED_EDIT: Replace uniform 384-token short attention with a compute-neutral 512/256-token stagewise split, retaining full context at layers 4 and 8 and all optimizer settings.

EVIDENCE: Uniform 384-token attention achieved 0.983618, while the opposite compute-neutral split—256 tokens early and 512 late—regressed to 0.984213; reversing that allocation directly tests whether local context is more valuable before the verified intermediate global-attention layer.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = 3 * long_window // 16
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        long_window = config.sequence_len
        early_short_window = long_window // 4
        late_short_window = long_window // 8
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                short_window = early_short_window if layer_idx < config.n_layer // 2 else late_short_window
                window_sizes.append((short_window, 0))
        window_sizes[-1] = (long_window, 0)
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=3/16 context
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 512 to 256
>>>>>>> REPLACE