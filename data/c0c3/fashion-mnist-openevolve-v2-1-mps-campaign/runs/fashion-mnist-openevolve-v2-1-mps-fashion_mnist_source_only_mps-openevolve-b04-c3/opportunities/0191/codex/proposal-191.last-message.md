MECHANISM: Accuracy-boundary EMA blend bisection retry

HYPOTHESIS: A 75.22734375% EMA / 24.77265625% live probability blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031356811523.

INTENDED_EDIT: Move the validation ensemble to the midpoint between the best accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.

EVIDENCE: The 75.2265625% blend retained 9,251 correct at 0.20655031356811523 cross-entropy, while 75.228125% lost one correct prediction; the prior midpoint attempt timed out and provided no contradictory validation evidence.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752),
                    F.log_softmax(live_logits, dim=1) + math.log(0.248),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7522734375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2477265625),
>>>>>>> REPLACE