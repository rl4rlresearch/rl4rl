MECHANISM: Projection-stack EMA alignment

HYPOTHESIS: Applying the proven 0.04 EMA rate to `view_fusion` as well as the classifier will exceed 9,330 correct predictions by reducing lag in the late-learned invariant/disagreement projection while retaining stable 0.02 averaging throughout the convolutional backbone.

INTENDED_EDIT: Treat `view_fusion` as part of the prediction head for parameter averaging, changing its EMA rate from 0.02 to 0.04.

EVIDENCE: A 0.04 EMA across both classifier layers reached 9,330 correct, whereas applying 0.04 only to the output layer reached 9,327; this indicates that faster averaging is most useful in feature-projection layers, directly motivating the same treatment for the adjacent zero-initialized fusion projection.

<<<<<<< SEARCH
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
=======
                        ema_rate = (
                            0.04
                            if (
                                name.startswith("view_fusion.")
                                or name.startswith("classifier.")
                            )
                            else 0.02
                        )
>>>>>>> REPLACE