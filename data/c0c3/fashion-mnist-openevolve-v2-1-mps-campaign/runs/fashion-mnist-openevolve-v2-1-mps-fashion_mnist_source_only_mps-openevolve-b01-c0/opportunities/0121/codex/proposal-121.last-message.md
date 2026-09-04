MECHANISM: Upper-side eighth-step live/EMA mixture refinement

HYPOTHESIS: A 50.68797035217285% live / 49.31202964782715% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest validated higher-live mixture.

EVIDENCE: The current 0.50687969970703125 live weight is best; the nearest tested mixtures on both sides were worse, but the higher-live neighbor had lower cross-entropy than the lower-live neighbor, making this untested upper midpoint the most informative refinement.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068797035217285),
                        ema_ensemble + math.log(0.4931202964782715),
>>>>>>> REPLACE