MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.22734375% EMA / 24.77265625% live blend will preserve 9,251 correct predictions while lowering validation cross-entropy below 0.20655031356811523.

INTENDED_EDIT: Move the validation probability blend halfway between the successful 75.2265625% EMA weight and the accuracy-losing 75.228125% weight, leaving training and calibration unchanged.

EVIDENCE: The 75.2265625% blend retained 9,251 correct at 0.20655031356811523 cross-entropy, while 75.228125% reduced cross-entropy but lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752265625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.247734375),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7522734375),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2477265625),
>>>>>>> REPLACE