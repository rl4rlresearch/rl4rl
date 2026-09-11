MECHANISM: Conservative live-weight continuation

HYPOTHESIS: A 50.4% live / 49.6% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2024912166595459.

INTENDED_EDIT: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.

EVIDENCE: Increasing live weight from 50.0% through 50.3% successively reduced cross-entropy while preserving 9,290 correct predictions; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next conservative step toward that boundary.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.503),
                        ema_ensemble + math.log(0.497),
=======
                        live_ensemble + math.log(0.504),
                        ema_ensemble + math.log(0.496),
>>>>>>> REPLACE