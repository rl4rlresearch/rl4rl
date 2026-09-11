MECHANISM: Argmax-preserving ensemble temperature calibration

HYPOTHESIS: Scaling evaluation logits by 0.99 will retain exactly 9,243 correct predictions while reducing validation cross-entropy below 0.20904547996520997.

INTENDED_EDIT: Apply a small softening factor only to inference-time flip-ensemble logits, leaving training, architecture, and class rankings unchanged.

EVIDENCE: Power-mean refinements around 0.5831695556640625 repeatedly retained 9,243 correct but no longer improved cross-entropy, motivating an orthogonal calibration adjustment whose positive scale mathematically preserves every non-tied argmax.

<<<<<<< SEARCH
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
=======
            logits = 0.99 * (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
        return logits
>>>>>>> REPLACE