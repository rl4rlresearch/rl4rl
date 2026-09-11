MECHANISM: Ultra-local EMA–endpoint blend calibration

HYPOTHESIS: A 75.1/24.9 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.20655155067443848.

INTENDED_EDIT: Reduce the live-model validation contribution from 25% to 24.9%, leaving training, runtime, parameters, flip ensembling, and logit scaling unchanged.

EVIDENCE: The 75/25 blend achieved 9,251 correct, while 77.5/22.5 lowered cross-entropy to 0.20653982543945312 but lost two correct predictions; this small lower-side step tests for tie-break improvement while minimizing the risk of crossing a prediction boundary.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.751),
                    F.log_softmax(live_logits, dim=1) + math.log(0.249),
                )
>>>>>>> REPLACE