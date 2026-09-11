MECHANISM: Conservative directional vertical-shift TTA extrapolation

HYPOTHESIS: An additional 1/8192 vertical-weight transfer will retain 9,348 correct predictions while lowering validation cross-entropy below 0.18770672912597655.

INTENDED_EDIT: Transfer 1/8192 ensemble weight from the first vertical crop to the second, preserving center, horizontal, and total weights.

EVIDENCE: The current additional 1/4096 transfer retained 9,348 correct and improved cross-entropy from 0.1877068977355957 to 0.18770672912597655; halving the next step conservatively tests whether this directional gain continues.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.312255859375,
            1.314697265625,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3121337890625,
            1.3148193359375,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE