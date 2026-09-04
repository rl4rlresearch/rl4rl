MECHANISM: Short-horizon second-moment adaptation

HYPOTHESIS: AdamW beta2=0.99 will exceed 9,268 correct predictions by adapting gradient-variance estimates faster during the fixed 522-step training run while retaining the best verified architecture and TTA.

INTENDED_EDIT: Restore Reference Design 3’s vertical-biased TTA and use a faster-decaying AdamW second-moment estimate.

EVIDENCE: Reference Design 3 achieved the best tied score with 9,268 correct; architecture, augmentation, and late-trajectory averaging changes regressed, motivating a clean optimization-dynamics test on the strongest verified representation.

<<<<<<< SEARCH
        logits = 0.38 * self._flip_average(images)
=======
        logits = 0.36 * self._flip_average(images)
>>>>>>> REPLACE

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.04
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
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE