MECHANISM: Lower-side sub-quantization live/EMA mixture refinement retry

HYPOTHESIS: A 50.6879669189453125% live / 49.3120330810546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Decrease the live-model ensemble weight from 0.50687969970703125 to 0.506879669189453125 and increase the EMA complement accordingly.

EVIDENCE: The current mixture is best at 9,290 correct and 0.20248969497680663 cross-entropy; the corresponding upper midpoint timed out twice, while this closest lower midpoint timed out only once, so retrying it resolves the most informative unverified local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.506879669189453125),
                        ema_ensemble + math.log(0.493120330810546875),
>>>>>>> REPLACE