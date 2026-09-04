MECHANISM: Mid-slope compute-neutral staged attention

HYPOTHESIS: A 640/1024/1408-token progression will lower val_bpb below 0.991835 while retaining at least 475M training tokens.

INTENDED_EDIT: Replace the overly steep 512/1024/1536 schedule with the midpoint between it and the best 768/1024/1280 schedule, preserving total short-attention FLOPs.

EVIDENCE: The 768/1024/1280 progression achieved the best val_bpb of 0.991835, while 512/1024/1536 regressed to 0.992198; their untested midpoint isolates whether the optimum lies between those slopes while keeping the successful 1024-token middle window.

<<<<<<< SEARCH
        # Keep the original total short-attention budget, but concentrate
        # context in the representation immediately handed to each long layer.
        short_windows = (
            long_window // 4,
            long_window // 2,
            3 * long_window // 4,
        )
=======
        # Preserve total short-attention FLOPs while testing the midpoint
        # between the best progression and the overly steep progression.
        short_windows = (
            5 * long_window // 16,
            long_window // 2,
            11 * long_window // 16,
        )
>>>>>>> REPLACE