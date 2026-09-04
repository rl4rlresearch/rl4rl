MECHANISM: Front-loaded hierarchical attention allocation

HYPOTHESIS: Retaining 128-token windows in the first half while shortening the three late local layers to 64 tokens will achieve `val_bpb < 0.982234` by preserving early contextualization and reallocating low-value late attention compute into additional training tokens.

INTENDED_EDIT: Keep the first pre-global layer at 256 tokens and the other early local layers at 128 tokens, while reducing all second-half local layers to 64 tokens.

EVIDENCE: A 256-token window only before the first full-context layer improved `val_bpb` from `0.982545` to `0.982234` while increasing throughput to 523.3M tokens; removing extra context from the later boundary helped, motivating a broader reduction of late local attention.

<<<<<<< SEARCH
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
                is_first_pre_global = next_char == "L" and layer_idx < config.n_layer // 2
                window = pre_global_window if is_first_pre_global else short_window
            window_sizes.append((window, 0))
=======
        long_window = config.sequence_len
        early_window = long_window // 16
        late_window = long_window // 32
        pre_global_window = long_window // 8
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window = long_window
            else:
                next_char = pattern[(layer_idx + 1) % len(pattern)]
                is_first_pre_global = next_char == "L" and layer_idx < config.n_layer // 2
                if is_first_pre_global:
                    window = pre_global_window
                elif layer_idx < config.n_layer // 2:
                    window = early_window
                else:
                    window = late_window
            window_sizes.append((window, 0))
>>>>>>> REPLACE