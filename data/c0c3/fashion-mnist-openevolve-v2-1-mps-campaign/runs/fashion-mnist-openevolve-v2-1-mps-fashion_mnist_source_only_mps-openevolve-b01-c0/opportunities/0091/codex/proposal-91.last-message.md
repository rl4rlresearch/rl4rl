MECHANISM: Binary search of the live/EMA decision boundary

HYPOTHESIS: A 50.65% live / 49.35% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249003524780274.

INTENDED_EDIT: Move the ensemble weight halfway between the successful 50.6% live mixture and the accuracy-losing 50.7% mixture.

EVIDENCE: The 50.6% live mixture retained 9,290 correct with cross-entropy 0.20249003524780274, while 50.7% reduced cross-entropy to 0.20248965530395507 but lost one correct prediction; their midpoint is the most informative boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.506),
                        ema_ensemble + math.log(0.494),
=======
                        live_ensemble + math.log(0.5065),
                        ema_ensemble + math.log(0.4935),
>>>>>>> REPLACE