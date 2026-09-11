MECHANISM: Upper-side local live/EMA cross-entropy refinement

HYPOTHESIS: A 50.688037109375% live / 49.311962890625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Move the ensemble weight halfway from the current best toward the nearby higher-live result.

EVIDENCE: The current 0.5068798828125 live weight achieved the best cross-entropy; the lower-side midpoint was worse, while the next tested higher-live weight also retained 9,290 correct, making the untested upper midpoint the most informative local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.50688037109375),
                        ema_ensemble + math.log(0.49311962890625),
>>>>>>> REPLACE