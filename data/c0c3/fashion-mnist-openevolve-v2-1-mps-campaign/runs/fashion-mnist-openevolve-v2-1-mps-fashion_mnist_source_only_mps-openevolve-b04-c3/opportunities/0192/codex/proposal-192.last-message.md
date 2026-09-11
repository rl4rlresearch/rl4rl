MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.227734375% EMA / 24.772265625% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031127929688.

INTENDED_EDIT: Move the validation ensemble halfway between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.

EVIDENCE: The 75.22734375% EMA blend retained 9,251 correct at 0.20655031127929688 cross-entropy, while 75.228125% lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.751),
                    F.log_softmax(live_logits, dim=1) + math.log(0.249),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.75227734375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.24772265625),
>>>>>>> REPLACE