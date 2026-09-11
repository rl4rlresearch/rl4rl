MECHANISM: Faster short-horizon second-moment adaptation

HYPOTHESIS: AdamW beta2=0.96 will exceed 9,278 correct predictions by continuing the verified improvement from beta2=0.99 through 0.98 to 0.97 while remaining stable over 522 optimizer steps.

INTENDED_EDIT: Restore the strongest vertical-biased TTA configuration and reduce AdamW beta2 to 0.96, preserving the architecture, augmentation, learning-rate schedule, and parameter count.

EVIDENCE: Beta2 reductions produced monotonic gains: 0.99 achieved 9,268 correct, 0.98 achieved 9,277, and 0.97 achieved 9,278 with the lowest cross-entropy of 0.2095485.

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
>>>>>>> REPLACE

<<<<<<< SEARCH
        ],
        lr=2.0e-3,
    )
=======
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
>>>>>>> REPLACE