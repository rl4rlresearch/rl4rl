MECHANISM: Fine-grained EMA–endpoint probability blend reweighting

HYPOTHESIS: Increasing the live-model contribution from 25% to 27.5% will exceed 9,251 correct predictions, or preserve 9,251 while lowering validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Reweight the validation ensemble from 75/25 to 72.5/27.5 EMA/live probabilities while retaining the verified architecture, training procedure, flip ensemble, and 1.048 calibration scale.

EVIDENCE: Adding 25% live-model probability to the 9,247-correct EMA predictor raised validation_correct to 9,251, demonstrating useful endpoint diversity; 27.5% is a conservative interpolation toward the unverified 30% proposal.

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