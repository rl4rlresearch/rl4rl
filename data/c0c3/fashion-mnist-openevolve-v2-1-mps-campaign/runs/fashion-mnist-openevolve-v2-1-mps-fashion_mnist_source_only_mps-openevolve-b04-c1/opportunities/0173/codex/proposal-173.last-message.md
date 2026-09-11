MECHANISM: Split hidden/output classifier EMA horizons

HYPOTHESIS: Using EMA rates of 0.04 for `classifier.1` and 0.03 for `classifier.4` will exceed 9,330 correct predictions by preserving faster tracking in the spatial projection while reducing the output layer’s harmful responsiveness.

INTENDED_EDIT: Retain the proven 0.04 EMA rate for the first classifier linear layer, moderate the final linear layer to 0.03, and keep all feature parameters at 0.02.

EVIDENCE: Applying 0.04 to the whole classifier reached 9,330, while applying 0.04 only to the final layer fell to 9,327; the split rates directly test whether moderating the output layer retains the successful projection-side benefit.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        if name.startswith("classifier.1."):
                            ema_rate = 0.04
                        elif name.startswith("classifier.4."):
                            ema_rate = 0.03
                        else:
                            ema_rate = 0.02
>>>>>>> REPLACE