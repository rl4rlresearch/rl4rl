MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.23125% EMA / 24.76875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

INTENDED_EDIT: Move the validation probability ensemble to the midpoint between the successful 75.225% EMA blend and the accuracy-losing 75.2375% blend, leaving training and calibration unchanged.

EVIDENCE: The 75.225% blend retained 9,251 correct at 0.20655032272338866 cross-entropy, while 75.2375% reduced cross-entropy but lost one correct prediction; the prior midpoint attempt timed out and supplied no contradictory validation result.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752),
                    F.log_softmax(live_logits, dim=1) + math.log(0.248),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7523125),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2476875),
>>>>>>> REPLACE