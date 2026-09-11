MECHANISM: EMA–endpoint probability blend reweighting

HYPOTHESIS: Reducing the live-model contribution from 25% to 20% will exceed 9,251 correct predictions, or tie while lowering cross-entropy below 0.20657955322265625.

INTENDED_EDIT: Change the validation ensemble from 75/25 to 80/20 EMA/live probabilities while preserving training and inference cost.

EVIDENCE: The stronger EMA alone reached 9,247 correct, while adding 25% of the weaker 9,243-correct live model raised the result to 9,251; a slightly more EMA-weighted blend tests whether 25% introduces excess endpoint error while retaining its useful diversity.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.80),
                    F.log_softmax(live_logits, dim=1) + math.log(0.20),
                )
>>>>>>> REPLACE