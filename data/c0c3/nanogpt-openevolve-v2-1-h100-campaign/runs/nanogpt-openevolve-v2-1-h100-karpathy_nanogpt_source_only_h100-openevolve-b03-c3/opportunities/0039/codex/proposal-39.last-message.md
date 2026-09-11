MECHANISM: Second-handoff-biased depth-adaptive staged attention

HYPOTHESIS: Using the milder 768/1024/1280 schedule for the first handoff and the steeper 640/1024/1408 schedule for the second will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.

INTENDED_EDIT: Replace the repeated 640/1024/1408 schedule with complementary compute-neutral schedules, assigning greater early-layer context to the first cycle and greater immediate-predecessor context to the second.

EVIDENCE: Repeated 704/1024/1344 achieved 0.991729, while assigning the steeper neighboring schedule to the first cycle regressed to 0.992249; reversing that depth allocation directly tests whether the second handoff benefits more from predecessor-focused context.

<<<<<<< SEARCH
        # Preserve total short-attention FLOPs while testing the midpoint
        # between the best progression and the overly steep progression.
        short_windows = (
            5 * long_window // 16,
            long_window // 2,
            11 * long_window // 16,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
=======
        # Preserve the best schedule's aggregate attention budget while testing
        # whether predecessor-focused context is more useful at the later handoff.
        first_cycle_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
        )
        second_cycle_windows = (
            5 * long_window // 16,
            long_window // 2,
            11 * long_window // 16,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            short_windows = first_cycle_windows if layer_idx < len(pattern) else second_cycle_windows
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
>>>>>>> REPLACE