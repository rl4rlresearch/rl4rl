MECHANISM: Fine-grained EMA–endpoint probability reweighting

HYPOTHESIS: Increasing the live-model contribution from 25% to 27.5% will exceed 9,251 correct predictions, or preserve 9,251 while lowering cross-entropy below 0.20655155601501465.

INTENDED_EDIT: Reweight the EMA/live ensemble to 72.5/27.5 and retain the best verified 1.0495 confidence calibration.

EVIDENCE: Adding 25% live probability to the stronger 9,247-correct EMA model produced 9,251 correct, demonstrating useful endpoint diversity; the prior 27.5% attempt timed out and therefore did not contradict this nearby interpolation.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.725),
                    F.log_softmax(live_logits, dim=1) + math.log(0.275),
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.03592 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE