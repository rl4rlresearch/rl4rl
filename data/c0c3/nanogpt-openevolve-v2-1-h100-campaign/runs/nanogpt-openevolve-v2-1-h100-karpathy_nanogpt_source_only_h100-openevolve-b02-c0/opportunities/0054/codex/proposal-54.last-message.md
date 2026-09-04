MECHANISM: Post-global attention-window compression

HYPOTHESIS: Using 424-token windows before the depth-4 global layer and 320-token windows afterward will reduce val_bpb below 0.983635 by preserving important early context while increasing training throughput.

INTENDED_EDIT: Keep the first three local layers at 424 tokens, shorten the three local layers after the intermediate full-context layer to 320 tokens, and retain full context at layers 4 and 8.

EVIDENCE: Uniform 320-token windows reached 0.983911, but shortening only the early layers while keeping later layers at 416 worsened val_bpb to 0.984264; this suggests local context is more valuable before global mixing and motivates testing the complementary allocation.

<<<<<<< SEARCH
        short_window = 53 * long_window // 256
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
=======
        short_window = 53 * long_window // 256
        post_global_window = 5 * long_window // 32
        window_sizes = []
        seen_global = False
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
                seen_global = True
            else:
                window = post_global_window if seen_global else short_window
                window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
>>>>>>> REPLACE