MECHANISM: Conservative binary search below the unresolved live/EMA boundary

HYPOTHESIS: A 50.687890625% live / 49.312109375% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20248970108032227.

INTENDED_EDIT: Increase the live-model ensemble weight from 0.506875 to 0.50687890625 and reduce the EMA weight complementarily.

EVIDENCE: The 50.6875% mixture retained 9,290 correct, while 50.6890625% lost one; two 50.68828125% trials timed out, so probing halfway between the verified-safe point and that unresolved midpoint is the most conservative informative continuation.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.506875),
                        ema_ensemble + math.log(0.493125),
=======
                        live_ensemble + math.log(0.50687890625),
                        ema_ensemble + math.log(0.49312109375),
>>>>>>> REPLACE