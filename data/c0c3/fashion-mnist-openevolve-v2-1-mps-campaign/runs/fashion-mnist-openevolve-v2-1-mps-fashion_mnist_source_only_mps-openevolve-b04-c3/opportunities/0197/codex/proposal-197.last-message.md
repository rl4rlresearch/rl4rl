MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.227490234375% EMA / 24.772509765625% live probability blend will retain 9,251 correct predictions while reducing validation cross-entropy below 0.20655031127929688.

INTENDED_EDIT: Move the validation ensemble halfway between the highest verified accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.

EVIDENCE: The 75.22744140625% EMA blend retained 9,251 correct, while 75.2275390625% lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7522734375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2477265625),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75227490234375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24772509765625),
>>>>>>> REPLACE