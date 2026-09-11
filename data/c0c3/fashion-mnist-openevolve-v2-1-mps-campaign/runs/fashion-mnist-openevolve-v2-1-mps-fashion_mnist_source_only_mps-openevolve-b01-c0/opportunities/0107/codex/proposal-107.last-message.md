MECHANISM: Conservative upper-side live/EMA cross-entropy refinement

HYPOTHESIS: A 50.68802490234375% live / 49.31197509765625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Increase the live-model ensemble weight from 0.5068798828125 to 0.5068802490234375 and reduce the EMA weight complementarily.

EVIDENCE: The nearby 0.506880126953125 mixture retained 9,290 correct and tied the best cross-entropy, while 0.50688037109375 timed out twice; probing their midpoint is the smallest informative upper-side refinement.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.5068802490234375),
                        ema_ensemble + math.log(0.4931197509765625),
>>>>>>> REPLACE