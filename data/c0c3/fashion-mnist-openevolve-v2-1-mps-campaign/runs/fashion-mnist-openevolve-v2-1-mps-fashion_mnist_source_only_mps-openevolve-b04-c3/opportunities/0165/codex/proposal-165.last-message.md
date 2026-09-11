MECHANISM: Lower-side EMA–endpoint blend refinement

HYPOTHESIS: A 76/24 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155067443848.

INTENDED_EDIT: Reduce the live-model validation contribution from 25% to 24% while retaining the verified training procedure, flip ensemble, EMA decay, and 1.04953 calibration.

EVIDENCE: The verified 75/25 blend achieved 9,251 correct, while raising live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy. The only lower-side test, 24.5%, timed out without producing contradictory performance evidence.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.76),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24),
                )
>>>>>>> REPLACE