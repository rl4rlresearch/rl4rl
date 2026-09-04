MECHANISM: Upper-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68797073364258% live / 49.31202926635742% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.

EVIDENCE: The current 0.50687969970703125 weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor produced lower cross-entropy than the lower-live neighbor, motivating refinement on the upper side.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068797073364258),
                        ema_ensemble + math.log(0.4931202926635742),
>>>>>>> REPLACE