MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.2265625% EMA / 24.7734375% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655032272338866.

INTENDED_EDIT: Move the validation probability blend halfway between the successful 75.225% EMA weight and the accuracy-losing 75.228125% weight, leaving training and calibration unchanged.

EVIDENCE: The 75.225% blend achieved 9,251 correct at 0.20655032272338866 cross-entropy, while 75.228125% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.247625),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752265625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.247734375),
>>>>>>> REPLACE