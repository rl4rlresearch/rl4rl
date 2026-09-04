MECHANISM: Conservative live-weight continuation

HYPOTHESIS: A 50.3% live / 49.7% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249161682128905.

INTENDED_EDIT: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.

EVIDENCE: Increasing live weight from 50.0% to 50.1% and then 50.2% preserved 9,290 correct predictions while successively reducing cross-entropy; 51.0% reduced it further but lost one correct prediction, motivating another conservative step toward that boundary.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.502),
                        ema_ensemble + math.log(0.498),
=======
                        live_ensemble + math.log(0.503),
                        ema_ensemble + math.log(0.497),
>>>>>>> REPLACE