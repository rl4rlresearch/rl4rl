MECHANISM: Conservative live-weight continuation retry

HYPOTHESIS: A 50.4% live / 49.6% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.2024912166595459.

INTENDED_EDIT: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.

EVIDENCE: Live weights from 50.0% through 50.3% successively reduced cross-entropy while preserving 9,290 correct predictions; the only 50.4% verification timed out, leaving this next interpolation unresolved.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.503),
                        ema_ensemble + math.log(0.497),
=======
                        live_ensemble + math.log(0.504),
                        ema_ensemble + math.log(0.496),
>>>>>>> REPLACE