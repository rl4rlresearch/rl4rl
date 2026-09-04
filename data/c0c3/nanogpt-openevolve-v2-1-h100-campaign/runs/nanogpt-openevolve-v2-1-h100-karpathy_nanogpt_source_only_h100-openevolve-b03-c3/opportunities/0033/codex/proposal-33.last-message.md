MECHANISM: Asymmetric predecessor-focused context allocation

HYPOTHESIS: Redistributing the middle short layer’s context to the immediate pre-long layer, while preserving the successful 768-token first window, will lower val_bpb below 0.991835 while retaining at least 475M training tokens.

INTENDED_EDIT: Change each compute-neutral short-window schedule from 768/1024/1280 to 768/896/1408 tokens.

EVIDENCE: The 768/1024/1280 schedule achieved the best val_bpb of 0.991835, while 512/1024/1536 regressed to 0.992198; holding the first window at 768 isolates whether additional pre-long context helps when the earliest layer is not starved.

<<<<<<< SEARCH
        # Preserve the original total short-attention budget while shifting
        # context toward the representation handed to each full-context layer.
        short_windows = (
            3 * long_window // 8,
            long_window // 2,
            5 * long_window // 8,
        )
=======
        # Preserve the original total short-attention budget and the successful
        # first-layer window while shifting context to the pre-long handoff.
        short_windows = (
            3 * long_window // 8,
            7 * long_window // 16,
            11 * long_window // 16,
        )
>>>>>>> REPLACE