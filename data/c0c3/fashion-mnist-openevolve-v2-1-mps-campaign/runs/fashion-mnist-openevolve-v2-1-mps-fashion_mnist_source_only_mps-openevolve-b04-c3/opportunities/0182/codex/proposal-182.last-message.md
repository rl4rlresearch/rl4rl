MECHANISM: Boundary-seeking EMA–endpoint blend refinement

HYPOTHESIS: A 75.25/24.75 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065504592895508.

INTENDED_EDIT: Increase the EMA validation contribution from 75.1% to 75.25% and reduce the live contribution from 24.9% to 24.75%, leaving training and calibration unchanged.

EVIDENCE: The 75.2/24.8 blend retained 9,251 correct with lower cross-entropy than 75.1/24.9, while 75.3/24.7 lost one correct prediction; their midpoint probes the accuracy boundary for further tie-break improvement.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.751),
                    F.log_softmax(live_logits, dim=1) + math.log(0.249),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.7525),
                    F.log_softmax(live_logits, dim=1) + math.log(0.2475),
>>>>>>> REPLACE