MECHANISM: Upper-side classifier EMA interpolation

HYPOTHESIS: A 0.0425 classifier EMA rate will exceed the 9,330-result by modestly reducing head lag without the instability observed at 0.05 and 0.06.

INTENDED_EDIT: Increase only the classifier-parameter EMA rate from 0.04 to 0.0425 while retaining the 0.02 backbone rate.

EVIDENCE: Classifier EMA peaked at 9,330 correct with 0.04, compared with 9,328 at 0.035, 9,329 at 0.05, and 9,327 at 0.06, motivating a narrow search immediately above the best observed rate.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.0425
                            if name.startswith("classifier.")
                            else 0.02
                        )
>>>>>>> REPLACE