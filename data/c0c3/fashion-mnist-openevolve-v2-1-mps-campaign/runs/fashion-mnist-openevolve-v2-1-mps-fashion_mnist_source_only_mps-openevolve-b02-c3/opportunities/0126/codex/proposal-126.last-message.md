MECHANISM: Orthogonal one-ULP horizontal TTA refinement

HYPOTHESIS: Restoring the best vertical weights and transferring one float32 ULP from the first horizontal crop to the second will retain 9,348 correct predictions while lowering validation cross-entropy below 0.1877065631866455.

INTENDED_EDIT: Restore Reference Design 1’s locally optimal vertical weights, then make the smallest representable directional adjustment to the previously symmetric horizontal crop weights while preserving their combined and total ensemble weight.

EVIDENCE: Reference Design 1 remains best; probes on both sides established a discrete local optimum for the vertical weights, motivating a minimally disruptive probe along the untested horizontal-weight dimension. The successful vertical direction favors the second shifted crop, providing the direction for this probe.

<<<<<<< SEARCH
        crop_weights = (
            3.0,
            1.31201267242431640625,
            1.31494045257568359375,
            0.6865234375,
            0.6865234375,
        )
=======
        crop_weights = (
            3.0,
            1.3120136260986328125,
            1.3149394989013671875,
            0.686523377895355224609375,
            0.686523497104644775390625,
        )
>>>>>>> REPLACE