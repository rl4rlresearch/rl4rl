MECHANISM: Boundary-seeking 0.58125-order probability power-mean flip ensemble

HYPOTHESIS: A 0.58125-order power mean will retain 9,243 correct predictions while reducing validation cross-entropy below 0.2090484.

INTENDED_EDIT: Replace arithmetic probability averaging with a 0.58125-order generalized mean, leaving training unchanged.

EVIDENCE: Order 0.5875 achieved 9,243 correct at 0.2090484 cross-entropy, while 0.575 reduced cross-entropy to 0.2090399 but lost one correct prediction; their midpoint probes the accuracy boundary while seeking better tie-break calibration.

<<<<<<< SEARCH
            logits = torch.logaddexp(
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            ) - math.log(2.0)
=======
            power = 0.58125
            logits = (
                torch.logaddexp(
                    power * F.log_softmax(logits, dim=1),
                    power * F.log_softmax(flipped_logits, dim=1),
                )
                - math.log(2.0)
            ) / power
>>>>>>> REPLACE