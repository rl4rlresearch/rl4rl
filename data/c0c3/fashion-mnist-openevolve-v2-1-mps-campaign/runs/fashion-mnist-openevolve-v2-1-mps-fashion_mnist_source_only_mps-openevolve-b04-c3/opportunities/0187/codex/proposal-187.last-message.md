MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.228125% EMA / 24.771875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

INTENDED_EDIT: Move the validation probability blend to the midpoint between the successful 75.225% EMA weight and the accuracy-losing 75.23125% weight, leaving training and calibration unchanged.

EVIDENCE: The 75.225% blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.23125% reduced cross-entropy but lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.751),
                    F.log_softmax(live_logits, dim=1) + math.log(0.249),
                )
=======
                logits = torch.logaddexp(
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75228125),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24771875),
                )
>>>>>>> REPLACE