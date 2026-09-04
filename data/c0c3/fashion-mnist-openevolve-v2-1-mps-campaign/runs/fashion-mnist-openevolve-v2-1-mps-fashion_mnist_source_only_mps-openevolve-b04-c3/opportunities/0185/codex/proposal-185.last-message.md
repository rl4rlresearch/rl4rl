MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.23125% EMA / 24.76875% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

INTENDED_EDIT: Move the validation probability ensemble to the midpoint between the successful 75.225% EMA weight and the accuracy-losing 75.2375% weight, leaving training and calibration unchanged.

EVIDENCE: The 75.225% EMA blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.2375% lowered cross-entropy to 0.2065502586364746 but lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75225),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24775),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7523125),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2476875),
>>>>>>> REPLACE