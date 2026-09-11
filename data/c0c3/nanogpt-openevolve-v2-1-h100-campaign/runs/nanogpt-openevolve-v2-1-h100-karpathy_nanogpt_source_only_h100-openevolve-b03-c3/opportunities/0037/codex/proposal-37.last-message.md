MECHANISM: Quadratically refined staged attention

HYPOTHESIS: A 720/1024/1328-token progression will lower val_bpb below 0.991729 while retaining at least 475M trained tokens.

INTENDED_EDIT: Shift 16 tokens of context from the final short layer to the first short layer, preserving the total short-attention FLOPs and all other proven components.

EVIDENCE: The current 704/1024/1344 schedule achieved 0.991729, outperforming the equally spaced neighboring schedules at 640/1024/1408 (0.991994) and 768/1024/1280 (0.991835); quadratic interpolation of these results places the local optimum near 718 tokens for the first window.

<<<<<<< SEARCH
        # Preserve total short-attention FLOPs while refining the interval
        # between the two strongest staged-window schedules.
        short_windows = (
            11 * long_window // 32,
            long_window // 2,
            21 * long_window // 32,
        )
=======
        # Preserve total short-attention FLOPs while probing the quadratic
        # optimum implied by the neighboring 640, 704, and 768-token results.
        short_windows = (
            45 * long_window // 128,
            long_window // 2,
            83 * long_window // 128,
        )
>>>>>>> REPLACE