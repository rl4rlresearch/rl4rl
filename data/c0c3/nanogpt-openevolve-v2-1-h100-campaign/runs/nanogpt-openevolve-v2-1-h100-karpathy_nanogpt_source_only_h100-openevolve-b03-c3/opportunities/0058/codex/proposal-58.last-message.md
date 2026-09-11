MECHANISM: Stronger pre-handoff attention-budget staging

HYPOTHESIS: Reallocating another 64 tokens from the earliest short-context layer to the pre-handoff layer, producing 640/1024/1408 windows, will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Preserve total short-attention FLOPs and the proven output-gate optimizer while increasing the staged window progression.

EVIDENCE: Moving from 768/1024/1280 at 0.991835 to 704/1024/1344 produced the best observed 0.991177; extending that same budget-neutral direction is the clearest untested continuation.

<<<<<<< SEARCH
        # Preserve total short-attention FLOPs with the best verified
        # progression toward each full-context handoff.
        short_windows = (
            11 * long_window // 32,
            long_window // 2,
            21 * long_window // 32,
        )
=======
        # Preserve total short-attention FLOPs while extending the successful
        # progression toward each full-context handoff by one 1/32 step.
        short_windows = (
            10 * long_window // 32,
            long_window // 2,
            22 * long_window // 32,
        )
>>>>>>> REPLACE