MECHANISM: Conservative live-weight continuation

HYPOTHESIS: A 50.5% live / 49.5% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249082069396973.

INTENDED_EDIT: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.

EVIDENCE: Increasing live weight from 50.0% through 50.4% successively reduced cross-entropy while preserving 9,290 correct predictions; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next conservative step toward that boundary.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.504),
                        ema_ensemble + math.log(0.496),
=======
                        live_ensemble + math.log(0.505),
                        ema_ensemble + math.log(0.495),
>>>>>>> REPLACE