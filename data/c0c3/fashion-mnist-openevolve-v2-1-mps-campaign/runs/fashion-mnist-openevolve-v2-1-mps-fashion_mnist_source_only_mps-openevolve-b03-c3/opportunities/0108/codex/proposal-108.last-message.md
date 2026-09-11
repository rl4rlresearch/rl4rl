MECHANISM: Faster short-horizon second-moment adaptation

HYPOTHESIS: AdamW beta2=0.95 will exceed 9,283 correct predictions by continuing the verified monotonic improvement from beta2=0.99 through 0.96 while remaining stable over 522 optimizer steps.

INTENDED_EDIT: Restore the strongest vertical-biased, distance-stratified TTA and reduce AdamW beta2 to 0.95 while preserving all other training and model settings.

EVIDENCE: Beta2 reductions improved validation from 9,268 correct at 0.99 to 9,277 at 0.98, 9,278 at 0.97, and 9,283 at 0.96, with cross-entropy decreasing monotonically to 0.2093681.

<<<<<<< SEARCH
                if radius == 1:
                    weight = (
                        0.09
                        if delta_y == 0 or delta_x == 0
                        else 0.045
                    )
                else:
                    weight = 0.00625
=======
                if radius == 1:
                    if delta_x == 0:
                        weight = 0.10
                    elif delta_y == 0:
                        weight = 0.08
                    else:
                        weight = 0.045
                else:
                    manhattan_distance = abs(delta_y) + abs(delta_x)
                    if manhattan_distance == 2:
                        weight = 0.009375
                    elif manhattan_distance == 3:
                        weight = 0.00625
                    else:
                        weight = 0.003125
>>>>>>> REPLACE

<<<<<<< SEARCH
        ],
        lr=2.0e-3,
    )
=======
        ],
        lr=2.0e-3,
        betas=(0.9, 0.95),
    )
>>>>>>> REPLACE