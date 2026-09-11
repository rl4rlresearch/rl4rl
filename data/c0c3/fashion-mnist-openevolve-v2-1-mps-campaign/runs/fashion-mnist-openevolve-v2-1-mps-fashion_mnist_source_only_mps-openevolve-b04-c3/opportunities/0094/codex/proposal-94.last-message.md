MECHANISM: Argmax-preserving ensemble temperature hardening

HYPOTHESIS: Scaling inference logits by 1.01 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Multiply only the inference-time flip-ensemble logits by 1.01, leaving training and class rankings unchanged.

EVIDENCE: Scaling logits by 0.99 preserved 9,243 correct but worsened cross-entropy to 0.20918512001037598, providing directional evidence that a symmetric hardening probe above 1.0 may improve calibration.

<<<<<<< SEARCH
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
=======
            logits = 1.01 * (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
>>>>>>> REPLACE