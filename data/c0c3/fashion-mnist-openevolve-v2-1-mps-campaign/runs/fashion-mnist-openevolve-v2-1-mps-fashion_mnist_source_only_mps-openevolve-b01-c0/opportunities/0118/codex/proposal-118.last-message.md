MECHANISM: Upper-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68797149658203% live / 49.31202850341797% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the unresolved higher-live midpoint.

EVIDENCE: The current 0.50687969970703125 live weight is the verified best; the nearest lower-live probe was worse, while the nearest higher-live probe timed out twice, making its untested halfway point the most informative refinement.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068797149658203),
                        ema_ensemble + math.log(0.4931202850341797),
>>>>>>> REPLACE