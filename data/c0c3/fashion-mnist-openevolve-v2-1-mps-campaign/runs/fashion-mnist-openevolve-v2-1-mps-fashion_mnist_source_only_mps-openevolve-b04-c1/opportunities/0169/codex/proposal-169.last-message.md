MECHANISM: Output-layer-specific EMA horizon

HYPOTHESIS: Applying the 0.04 EMA rate only to the final linear layer will exceed 9,330 correct predictions by tracking label-smoothing and dropout annealing at the logits while giving the classifier’s spatial feature projection the more stable 0.02 backbone horizon.

INTENDED_EDIT: Restrict the faster classifier EMA from the entire classifier to `classifier.4`, leaving all other learned parameters at 0.02.

EVIDENCE: Raising the whole classifier’s EMA rate from 0.02 to 0.04 improved correctness from 9,328 to 9,330, while increasing it to 0.05 and 0.06 reduced correctness; the final linear layer is the component most directly exposed to the annealed dropout and label-smoothed objective.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.4.")
                            else 0.02
                        )
>>>>>>> REPLACE