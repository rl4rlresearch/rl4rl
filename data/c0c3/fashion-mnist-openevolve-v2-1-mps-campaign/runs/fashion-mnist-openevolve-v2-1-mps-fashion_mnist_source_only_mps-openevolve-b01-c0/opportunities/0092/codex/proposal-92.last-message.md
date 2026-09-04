MECHANISM: Binary search of the live/EMA accuracy boundary

HYPOTHESIS: A 50.675% live / 49.325% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024898406982422.

INTENDED_EDIT: Move the ensemble weight halfway between the successful 50.65% live mixture and the accuracy-losing 50.7% mixture.

EVIDENCE: The 50.65% live mixture retained 9,290 correct at 0.2024898406982422 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative next boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5065),
                        ema_ensemble + math.log(0.4935),
=======
                        live_ensemble + math.log(0.50675),
                        ema_ensemble + math.log(0.49325),
>>>>>>> REPLACE