MECHANISM: Lower-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68796844482422% live / 49.31203155517578% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest validated lower-live mixture.

EVIDENCE: The current 0.50687969970703125 live weight is the verified best; the adjacent lower-live weight 0.506879669189453125 was worse, while the corresponding upper-side quarter-step was also worse, making the untested lower-side quarter-step the most informative local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068796844482422),
                        ema_ensemble + math.log(0.4931203155517578),
>>>>>>> REPLACE