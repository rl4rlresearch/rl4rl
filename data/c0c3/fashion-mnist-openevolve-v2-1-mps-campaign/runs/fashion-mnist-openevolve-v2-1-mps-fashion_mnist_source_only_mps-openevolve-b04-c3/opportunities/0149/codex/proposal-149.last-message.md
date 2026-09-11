MECHANISM: Lower-side EMA–endpoint blend refinement

HYPOTHESIS: A 76/24 EMA/live probability blend will exceed 9,251 correct predictions, or retain 9,251 while reducing cross-entropy below 0.20655155601501465.

INTENDED_EDIT: Reduce the live-model contribution from 25% to 24% while preserving the verified architecture, training procedure, flip ensemble, and 1.0495 calibration.

EVIDENCE: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy; this motivates a conservative test on the lower-live-weight side. Prior 23.75% attempts timed out and provide no contradictory validation result.

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