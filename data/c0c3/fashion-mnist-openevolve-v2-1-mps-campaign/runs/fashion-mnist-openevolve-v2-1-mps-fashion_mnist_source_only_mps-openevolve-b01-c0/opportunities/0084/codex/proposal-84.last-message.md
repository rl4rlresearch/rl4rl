MECHANISM: Conservative live-weight continuation retry

HYPOTHESIS: A 50.3% live / 49.7% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249161682128905.

INTENDED_EDIT: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.

EVIDENCE: Increasing live weight from 50.0% through 50.2% successively reduced cross-entropy while preserving 9,290 correct predictions; the 50.3% trial timed out, so its objective result remains unresolved.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.502),
                        ema_ensemble + math.log(0.498),
=======
                        live_ensemble + math.log(0.503),
                        ema_ensemble + math.log(0.497),
>>>>>>> REPLACE