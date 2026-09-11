MECHANISM: Lower-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.687969970703125% live / 49.312030029296875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest tested lower-live mixture.

EVIDENCE: The current 0.5068797607421875 mixture achieved the best result; both adjacent tested mixtures were worse, and the upper-side midpoint was also worse, making the corresponding untested lower-side midpoint the most informative remaining probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068797607421875),
                        ema_ensemble + math.log(0.4931202392578125),
=======
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
>>>>>>> REPLACE