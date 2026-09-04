MECHANISM: Conservative lower-side EMA–endpoint blend refinement

HYPOTHESIS: A 75.25/24.75 EMA/live probability blend will exceed 9,251 correct predictions, or preserve 9,251 while reducing validation cross-entropy below 0.20655155067443848.

INTENDED_EDIT: Reduce the live-model validation contribution from 25% to 24.75%, increase the EMA contribution correspondingly, and use the best verified 1.04953 calibration.

EVIDENCE: The verified 75/25 blend achieved 9,251 correct, while increasing live weight to 27.5% reduced correctness to 9,249 and worsened cross-entropy; unsuccessful 24–24.5% runs produced no contradictory validation measurement, motivating a conservative lower-side interpolation.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
            else:
                logits = ema_logits
            logits = 1.0495 * logits
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7525),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2475),
                )
            else:
                logits = ema_logits
            logits = 1.04953 * logits
>>>>>>> REPLACE