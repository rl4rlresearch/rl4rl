MECHANISM: Conservative directional vertical-shift TTA extrapolation

HYPOTHESIS: Moving another 1/2048 ensemble weight from the downward-shifted crop to the upward-shifted crop will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877068977355957.

INTENDED_EDIT: Restore the best verified ensemble mean and horizontal weights, then increase the successful vertical directional asymmetry by half the previously tested step while preserving total ensemble weight.

EVIDENCE: A 1/1024 directional redistribution retained 9,348 correct and improved cross-entropy from approximately 0.187707568 to 0.187706898; a conservative half-step extrapolation directly tests whether that gain continues before risking a larger decision-boundary crossing.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.3134918212890625,
            1.3134918212890625,
            0.6865081787109375,
            0.6865081787109375,
        )
=======
        crop_weights = (
            3.0,
            1.31201171875,
            1.31494140625,
            0.6865234375,
            0.6865234375,
        )
>>>>>>> REPLACE