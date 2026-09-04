MECHANISM: Lower-side classifier EMA interpolation

HYPOTHESIS: A classifier EMA rate of 0.035 will exceed 9,330 correct predictions by retaining most of the reduced head lag gained at 0.04 while adding stability suggested by the regression at 0.05 and 0.06.

INTENDED_EDIT: Reduce the EMA update rate for all classifier parameters from 0.04 to 0.035 while retaining the proven 0.02 feature-parameter rate.

EVIDENCE: Classifier-only EMA improved from 9,328 correct at 0.02 to 9,330 at 0.04, then declined to 9,329 at 0.05 and 9,327 at 0.06; testing 0.035 brackets the unexplored lower side of the apparent local optimum.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.035
                            if name.startswith("classifier.")
                            else 0.02
                        )
>>>>>>> REPLACE