MECHANISM: Local live/EMA cross-entropy refinement

HYPOTHESIS: A 50.68798828125% live / 49.31201171875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.2024896987915039.

INTENDED_EDIT: Move the ensemble weight halfway from the current best 0.50687890625 live weight toward the accuracy-safe but cross-entropy-worse 0.506880859375 result.

EVIDENCE: The current mixture achieved the best score at 9,290 correct and 0.2024896987915039 cross-entropy, while the nearby higher-live mixture retained 9,290 correct but raised cross-entropy; their midpoint is the most informative probe of the apparent local optimum.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687890625),
                        ema_ensemble + math.log(0.49312109375),
=======
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
>>>>>>> REPLACE