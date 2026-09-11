MECHANISM: Ultra-local EMA–endpoint blend refinement

HYPOTHESIS: A 75.2/24.8 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065509983062744.

INTENDED_EDIT: Increase the EMA validation contribution from 75% to 75.2%, reduce the live contribution to 24.8%, and use the calibration verified with the improved 75.1/24.9 blend.

EVIDENCE: The 75.1/24.9 blend preserved 9,251 correct while improving cross-entropy to 0.2065509983062744; 77.5/22.5 further lowered cross-entropy but lost two correct predictions, motivating a small continuation toward EMA rather than a large step.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
            else:
                logits = ema_logits
            logits = 1.0496144 * logits
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752),
                    F.log_softmax(live_logits, dim=1) + math.log(0.248),
                )
            else:
                logits = ema_logits
            logits = 1.04953 * logits
>>>>>>> REPLACE