MECHANISM: Hidden-projection-specific EMA horizon

HYPOTHESIS: Applying the 0.04 EMA rate only to `classifier.1` will exceed 9,330 correct predictions by retaining faster tracking for the spatial feature projection while avoiding the harmful output-layer-specific averaging change.

INTENDED_EDIT: Use a 0.04 EMA update rate for the classifier’s first linear layer and 0.02 for every other parameter, including the final linear layer.

EVIDENCE: Faster EMA across both classifier layers reached 9,330, whereas applying it only to `classifier.4` fell to 9,327; this isolates the first linear projection as the likely source of the improvement and removes the output-layer change that was detrimental in isolation.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.1.")
                            else 0.02
                        )
>>>>>>> REPLACE