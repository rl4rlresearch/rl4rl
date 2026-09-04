MECHANISM: Binary search of the live/EMA accuracy boundary

HYPOTHESIS: A 50.68828125% live / 49.31171875% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248970108032227.

INTENDED_EDIT: Move the ensemble weight halfway between the successful 50.6875% live mixture and the accuracy-losing 50.6890625% mixture, retaining the verified temperature.

EVIDENCE: The 50.6875% mixture retained 9,290 correct, while 50.6890625% lost one correct despite slightly lower cross-entropy; their midpoint is the most informative unresolved boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.506875),
                        ema_ensemble + math.log(0.493125),
=======
                        live_ensemble + math.log(0.5068828125),
                        ema_ensemble + math.log(0.4931171875),
>>>>>>> REPLACE