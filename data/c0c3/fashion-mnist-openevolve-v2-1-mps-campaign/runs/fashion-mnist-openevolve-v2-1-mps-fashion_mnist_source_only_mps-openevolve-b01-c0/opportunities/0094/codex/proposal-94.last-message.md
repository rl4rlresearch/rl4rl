MECHANISM: Binary search of the live/EMA accuracy boundary

HYPOTHESIS: A 50.6875% live / 49.3125% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248974838256836.

INTENDED_EDIT: Move the ensemble weight halfway between the successful 50.675% live mixture and the accuracy-losing 50.7% mixture, retaining the verified temperature.

EVIDENCE: The 50.675% live mixture retained 9,290 correct at 0.20248974838256836 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative unresolved boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50675),
                        ema_ensemble + math.log(0.49325),
=======
                        live_ensemble + math.log(0.506875),
                        ema_ensemble + math.log(0.493125),
>>>>>>> REPLACE