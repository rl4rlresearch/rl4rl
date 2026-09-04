MECHANISM: Conservative live-weight continuation

HYPOTHESIS: A 50.2% live / 49.8% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.20249202919006348.

INTENDED_EDIT: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the best verified temperature.

EVIDENCE: Moving from 50.0% to 50.1% live preserved 9,290 correct predictions and reduced cross-entropy, while 51.0% live reduced cross-entropy further but crossed an accuracy boundary and lost one correct prediction.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.501),
                        ema_ensemble + math.log(0.499),
=======
                        live_ensemble + math.log(0.502),
                        ema_ensemble + math.log(0.498),
>>>>>>> REPLACE