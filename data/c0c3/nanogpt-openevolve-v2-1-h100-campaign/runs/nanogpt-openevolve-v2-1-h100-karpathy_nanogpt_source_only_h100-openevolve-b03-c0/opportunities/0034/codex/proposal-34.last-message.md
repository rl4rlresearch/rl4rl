MECHANISM: Hierarchical local-to-global attention allocation

HYPOTHESIS: Using 128-token windows for four local layers while retaining 256-token windows immediately before each full-context layer will achieve `val_bpb < 0.982793` by preserving high-value local aggregation while recovering most of the 128-token throughput gain.

INTENDED_EDIT: Shorten four local-attention layers to 128 tokens and keep the two local layers directly preceding full-context attention at 256 tokens.

EVIDENCE: Uniform 128-token windows increased throughput from 516.6M to 520.8M tokens with only a `0.000013` regression versus 256 tokens; concentrating 256-token windows at the local-to-global boundaries tests whether that small quality advantage can be retained efficiently.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
=======
        long_window = config.sequence_len
        short_window = long_window // 16
        pre_global_window = long_window // 8
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window = long_window
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                window = pre_global_window if next_char == "L" else short_window
            window_sizes.append((window, 0))
>>>>>>> REPLACE