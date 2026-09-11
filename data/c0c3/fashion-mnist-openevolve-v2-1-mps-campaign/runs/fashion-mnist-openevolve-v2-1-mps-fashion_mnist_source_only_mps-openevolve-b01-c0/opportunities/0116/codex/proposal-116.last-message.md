MECHANISM: Lower-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.6879669189453125% live / 49.3120330810546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest tested lower-live mixture.

EVIDENCE: The current 0.50687969970703125 live weight achieved the best verified cross-entropy; the nearest lower-live mixture was worse, while the corresponding upper midpoint timed out twice, making the untested lower midpoint the most informative remaining local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.506879669189453125),
                        ema_ensemble + math.log(0.493120330810546875),
>>>>>>> REPLACE