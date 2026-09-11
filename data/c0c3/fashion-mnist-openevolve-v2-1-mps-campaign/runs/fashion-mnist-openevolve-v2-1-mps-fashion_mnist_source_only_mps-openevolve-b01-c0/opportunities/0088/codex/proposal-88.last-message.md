MECHANISM: Conservative live-weight continuation retry

HYPOTHESIS: A 50.5% live / 49.5% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249082069396973.

INTENDED_EDIT: Shift 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.

EVIDENCE: The timed-out 50.5% trial reported 9,290 correct predictions and cross-entropy of 0.202490425491333, improving on the accepted 50.4% result; retrying can establish that gain as a valid design.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.504),
                        ema_ensemble + math.log(0.496),
=======
                        live_ensemble + math.log(0.505),
                        ema_ensemble + math.log(0.495),
>>>>>>> REPLACE