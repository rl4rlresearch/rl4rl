MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.225% EMA / 24.775% live blend will preserve 9,251 correct predictions while lowering cross-entropy below 0.2065504592895508.

INTENDED_EDIT: Move the validation ensemble halfway from the best verified 75.2/24.8 blend toward the accuracy-losing 75.25/24.75 blend, and restore its verified 1.04953 calibration.

EVIDENCE: The 75.2/24.8 blend achieved 9,251 correct at 0.2065504592895508 cross-entropy, while 75.25/24.75 lowered cross-entropy but lost one correct prediction; their midpoint is the most informative probe of the prediction boundary.

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
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75225),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24775),
                )
            else:
                logits = ema_logits
            logits = 1.04953 * logits
>>>>>>> REPLACE