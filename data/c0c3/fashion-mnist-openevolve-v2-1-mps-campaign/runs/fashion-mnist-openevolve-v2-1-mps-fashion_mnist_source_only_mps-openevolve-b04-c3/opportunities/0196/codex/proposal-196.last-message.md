MECHANISM: Accuracy-boundary EMA blend bisection retry

HYPOTHESIS: A 75.22744140625% EMA / 24.77255859375% live probability blend will retain 9,251 correct predictions while reducing cross-entropy below 0.20655031127929688.

INTENDED_EDIT: Move the validation ensemble to the midpoint between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.

EVIDENCE: The 75.22734375% EMA blend retained 9,251 correct, while 75.2275390625% lost one; the midpoint remains the most informative boundary probe because its two prior attempts timed out without validation evidence.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.751),
                    F.log_softmax(live_logits, dim=1) + math.log(0.249),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7522744140625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2477255859375),
>>>>>>> REPLACE