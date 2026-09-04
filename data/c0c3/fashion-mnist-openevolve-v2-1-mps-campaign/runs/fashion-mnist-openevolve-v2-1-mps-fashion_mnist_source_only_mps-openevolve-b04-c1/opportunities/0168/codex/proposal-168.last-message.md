MECHANISM: Midpoint classifier-only EMA horizon

HYPOTHESIS: A classifier EMA rate of 0.05 will exceed 9,330 correct predictions by balancing the reduced head-parameter lag achieved at 0.04 against the excessive responsiveness observed at 0.06.

INTENDED_EDIT: Increase only the classifier’s second-half EMA update rate from 0.04 to 0.05, leaving feature-parameter EMA and all training dynamics unchanged.

EVIDENCE: Classifier-only EMA at 0.04 improved correctness to 9,330, while 0.06 fell to 9,327; testing their midpoint directly brackets the apparent local optimum without disturbing the proven 0.02 feature-parameter horizon.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.05
                            if name.startswith("classifier.")
                            else 0.02
                        )
>>>>>>> REPLACE