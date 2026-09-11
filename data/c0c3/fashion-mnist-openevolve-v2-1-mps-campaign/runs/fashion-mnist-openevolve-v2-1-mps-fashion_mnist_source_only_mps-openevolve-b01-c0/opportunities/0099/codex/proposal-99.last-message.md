MECHANISM: Binary-search retry at the live/EMA accuracy boundary

HYPOTHESIS: A 50.68828125% live / 49.31171875% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248970108032227.

INTENDED_EDIT: Retry the unresolved midpoint between the successful 50.6875% mixture and the accuracy-losing 50.6890625% mixture.

EVIDENCE: The only 50.68828125% verification timed out; neighboring evidence brackets the accuracy boundary between 50.6875% live, which retained 9,290 correct, and 50.6890625%, which lost one correct.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.506875),
                        ema_ensemble + math.log(0.493125),
=======
                        live_ensemble + math.log(0.5068828125),
                        ema_ensemble + math.log(0.4931171875),
>>>>>>> REPLACE