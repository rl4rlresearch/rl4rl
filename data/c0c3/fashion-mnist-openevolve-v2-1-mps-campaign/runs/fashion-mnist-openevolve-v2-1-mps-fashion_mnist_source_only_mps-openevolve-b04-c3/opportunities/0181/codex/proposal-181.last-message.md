MECHANISM: Ultra-local EMA–endpoint blend continuation

HYPOTHESIS: A 75.3/24.7 EMA/live probability blend will preserve all 9,251 correct predictions while reducing validation cross-entropy below 0.2065504592895508.

INTENDED_EDIT: Increase the EMA validation contribution from 75.2% to 75.3% and reduce the live contribution from 24.8% to 24.7%, leaving training and calibration unchanged.

EVIDENCE: Successive moves from 75.0/25.0 to 75.1/24.9 and 75.2/24.8 each preserved 9,251 correct while lowering cross-entropy; the next equally sized step directly tests whether that improvement continues before the known accuracy loss at 77.5/22.5.

<<<<<<< SEARCH
                    F.log_softmax(ema_logits, dim=1) + math.log(0.752),
                    F.log_softmax(live_logits, dim=1) + math.log(0.248),
=======
                    F.log_softmax(ema_logits, dim=1) + math.log(0.753),
                    F.log_softmax(live_logits, dim=1) + math.log(0.247),
>>>>>>> REPLACE