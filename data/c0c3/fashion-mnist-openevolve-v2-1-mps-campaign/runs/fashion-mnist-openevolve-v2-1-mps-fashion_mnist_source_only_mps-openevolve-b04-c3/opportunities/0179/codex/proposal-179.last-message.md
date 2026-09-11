MECHANISM: Ultra-local EMA–endpoint blend continuation

HYPOTHESIS: A 75.2/24.8 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065509983062744.

INTENDED_EDIT: Increase the EMA validation contribution from 75.0% to 75.2%, reduce the live contribution to 24.8%, and use the 1.04953 calibration associated with the best verified blend.

EVIDENCE: Moving from 75/25 to 75.1/24.9 preserved 9,251 correct and reduced cross-entropy from 0.20655155067443848 to 0.2065509983062744; the prior 75.2/24.8 attempt timed out without contradictory validation evidence.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75),
                    F.log_softmax(live_logits, dim=1) + math.log(0.25),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752),
                    F.log_softmax(live_logits, dim=1) + math.log(0.248),
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
            logits = 1.0495 * logits
=======
            logits = 1.04953 * logits
>>>>>>> REPLACE