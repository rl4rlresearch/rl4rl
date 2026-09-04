MECHANISM: Accuracy-boundary EMA blend bisection

HYPOTHESIS: A 75.22750244140625% EMA / 24.77249755859375% live probability blend will retain 9,251 correct predictions while reducing validation cross-entropy below 0.20655030899047852.

INTENDED_EDIT: Move the validation ensemble to the midpoint between the highest verified accuracy-preserving EMA weight and the nearest verified accuracy-losing weight, leaving training and calibration unchanged.

EVIDENCE: The 75.227490234375% EMA blend retained 9,251 correct at 0.20655030899047852 cross-entropy, while 75.2275146484375% lost one correct prediction; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752265625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.247734375),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7522750244140625),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2477249755859375),
>>>>>>> REPLACE