MECHANISM: Lower-side EMA–endpoint blend refinement

HYPOTHESIS: A 75.5/24.5 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155601501465.

INTENDED_EDIT: Reduce the live-model validation contribution from 25% to 24.5% while retaining the verified architecture, training procedure, flip ensemble, EMA decay, and 1.0495 calibration.

EVIDENCE: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy. The unmeasured lower side remains the most direct local refinement; 24.5% tests it conservatively.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.755),
                    F.log_softmax(live_logits, dim=1) + math.log(0.245),
                )
>>>>>>> REPLACE