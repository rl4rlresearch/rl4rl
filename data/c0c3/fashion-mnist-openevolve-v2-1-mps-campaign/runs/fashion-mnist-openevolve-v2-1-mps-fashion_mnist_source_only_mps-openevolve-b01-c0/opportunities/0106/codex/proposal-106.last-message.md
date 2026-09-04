MECHANISM: Conservative upper-side live/EMA cross-entropy refinement

HYPOTHESIS: A 50.6880126953125% live / 49.3119873046875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Increase the live-model ensemble weight from 0.5068798828125 to 0.506880126953125 and reduce the EMA weight complementarily.

EVIDENCE: The current weight achieved the best verified result, while the adjacent upper midpoint timed out twice; probing halfway toward that unresolved point is the smallest informative upper-side refinement.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.506880126953125),
                        ema_ensemble + math.log(0.493119873046875),
>>>>>>> REPLACE