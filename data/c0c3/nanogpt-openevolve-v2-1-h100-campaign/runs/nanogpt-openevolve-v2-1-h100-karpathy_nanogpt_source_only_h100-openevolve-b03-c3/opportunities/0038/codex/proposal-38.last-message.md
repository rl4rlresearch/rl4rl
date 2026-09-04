MECHANISM: First-handoff-biased depth-adaptive staged attention

HYPOTHESIS: Using the steeper 640/1024/1408 schedule for the first short-to-long handoff and the milder 768/1024/1280 schedule for the second will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.

INTENDED_EDIT: Replace the repeated short-window schedule with complementary, compute-neutral schedules that allocate more predecessor context to the empirically more informative first handoff.

EVIDENCE: Repeating the midpoint 704/1024/1344 schedule achieved the best val_bpb of 0.991729, while its aligned neighbors 640/1024/1408 and 768/1024/1280 reached 0.991994 and 0.991835; using those neighbors in separate depth cycles preserves the midpoint’s aggregate attention budget while testing the layer dependence suggested by the stronger first-handoff gating evidence.

<<<<<<< SEARCH
        # Preserve the original total short-attention budget while shifting
        # context toward the representation handed to each full-context layer.
        short_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
=======
        # Preserve the total attention budget while emphasizing the first
        # short-to-long handoff and using a milder progression in the final cycle.
        short_windows_by_cycle = (
            (
                5 * long_window // 16,
                long_window // 2,
                11 * long_window // 16,
            ),
            (
                3 * long_window // 8,
                long_window // 2,
                5 * long_window // 8,
            ),
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            cycle_idx = (layer_idx // len(pattern)) % len(short_windows_by_cycle)
            char = pattern[pattern_idx]
            short_windows = short_windows_by_cycle[cycle_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
>>>>>>> REPLACE