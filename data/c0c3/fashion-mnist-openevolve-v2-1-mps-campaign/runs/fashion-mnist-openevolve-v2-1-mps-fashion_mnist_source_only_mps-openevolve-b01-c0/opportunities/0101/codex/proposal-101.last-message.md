MECHANISM: Conservative binary search below the unresolved live/EMA boundary

HYPOTHESIS: A 50.6880859375% live / 49.3119140625% EMA probability mixture will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024896987915039.

INTENDED_EDIT: Increase the live-model ensemble weight from 0.50687890625 to 0.506880859375 and reduce the EMA weight complementarily.

EVIDENCE: The 50.687890625% live mixture retained 9,290 correct, while 50.6890625% lost one; two trials at the intervening 50.68828125% point timed out, so probing halfway between the verified-safe point and that unresolved point is the most conservative informative continuation.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687890625),
                        ema_ensemble + math.log(0.49312109375),
=======
                        live_ensemble + math.log(0.506880859375),
                        ema_ensemble + math.log(0.493119140625),
>>>>>>> REPLACE