MECHANISM: Faster terminal classifier EMA

HYPOTHESIS: Increasing only the classifier EMA update rate from 0.04 to 0.06 will exceed 9,330 correct predictions by further reducing head-parameter lag while retaining the proven 0.02 averaging horizon for feature extraction.

INTENDED_EDIT: Shorten the classifier’s second-half EMA horizon from roughly 25 to 17 optimizer steps without changing training dynamics, backbone averaging, or BatchNorm-buffer handling.

EVIDENCE: A classifier-specific 0.04 rate with the backbone held at 0.02 improved correctness from 9,328 to 9,330, whereas changing the EMA rate globally was weaker; this motivates continuing in the successful classifier-only direction with a moderate increase.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.06
                            if name.startswith("classifier.")
                            else 0.02
                        )
>>>>>>> REPLACE