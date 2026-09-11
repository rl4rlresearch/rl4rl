MECHANISM: Layer-specific classifier EMA tracking

HYPOTHESIS: Using a 0.05 EMA rate only for the final linear layer while retaining 0.04 for the hidden classifier projection will exceed 9,330 correct predictions by reducing output-layer lag without weakening the averaging of learned spatial features.

INTENDED_EDIT: Split classifier EMA rates: 0.05 for `classifier.4` parameters, 0.04 for the rest of the classifier, and 0.02 elsewhere.

EVIDENCE: Applying 0.05 to the entire classifier reached 9,329 versus 9,330 at 0.04, while applying 0.04 only to the output layer reached 9,327; this suggests the hidden projection benefits from the proven 0.04 rate while leaving the rapidly adapting output layer’s preferred rate unresolved.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
                        averaged.lerp_(value.detach(), ema_rate)
=======
                        if name.startswith("classifier.4."):
                            ema_rate = 0.05
                        elif name.startswith("classifier."):
                            ema_rate = 0.04
                        else:
                            ema_rate = 0.02
                        averaged.lerp_(value.detach(), ema_rate)
>>>>>>> REPLACE