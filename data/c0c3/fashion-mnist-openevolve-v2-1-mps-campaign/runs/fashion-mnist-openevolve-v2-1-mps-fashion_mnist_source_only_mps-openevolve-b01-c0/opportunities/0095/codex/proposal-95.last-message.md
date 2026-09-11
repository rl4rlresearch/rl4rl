MECHANISM: Binary search of the live/EMA accuracy boundary

HYPOTHESIS: A 50.69375% live / 49.30625% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248970108032227.

INTENDED_EDIT: Move the ensemble weight halfway between the successful 50.6875% live mixture and the accuracy-losing 50.7% mixture while retaining the verified temperature.

EVIDENCE: The 50.6875% live mixture retained 9,290 correct at 0.20248970108032227 cross-entropy, while 50.7% lowered cross-entropy but lost one correct prediction; their midpoint is the most informative unresolved boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.506875),
                        ema_ensemble + math.log(0.493125),
=======
                        live_ensemble + math.log(0.5069375),
                        ema_ensemble + math.log(0.4930625),
>>>>>>> REPLACE