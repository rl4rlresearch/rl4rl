MECHANISM: Axis-aligned translation augmentation

HYPOTHESIS: Matching training augmentation to the successful 0.10 vertical/0.08 horizontal validation weighting will exceed 9,268 correct predictions by learning stronger invariance along the empirically favored axis.

INTENDED_EDIT: Increase each radius-1 vertical training-translation weight from 0.09 to 0.10 and decrease each horizontal weight to 0.08, preserving the augmentation distribution’s total weight and all other settings.

EVIDENCE: Moderate vertical-biased TTA retained 9,268 correct and achieved the best cross-entropy, while horizontal bias worsened cross-entropy and stronger vertical bias lost five predictions; this motivates aligning training with the successful moderate bias.

<<<<<<< SEARCH
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
=======
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.10000, 0.04500, 0.00625,
            0.00625, 0.08000, 0.36000, 0.08000, 0.00625,
            0.00625, 0.04500, 0.10000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
>>>>>>> REPLACE