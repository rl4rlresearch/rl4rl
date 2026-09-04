MECHANISM: Lower-side EMA–endpoint blend interpolation

HYPOTHESIS: A 76.25/23.75 EMA/live probability blend will exceed 9,251 correct predictions, or retain 9,251 while reducing cross-entropy below 0.20655155601501465.

INTENDED_EDIT: Reduce the live-model ensemble weight from 25% to 23.75% while preserving the verified architecture, training procedure, flip ensemble, and 1.0495 calibration.

EVIDENCE: The 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249; the prior 23.75% attempt timed out and therefore provides no contradictory validation evidence.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2375),
>>>>>>> REPLACE