MECHANISM: Conservative live-weight continuation

HYPOTHESIS: A 50.6% live / 49.4% EMA probability mixture will preserve all 9,290 correct predictions while reducing validation cross-entropy below 0.202490425491333.

INTENDED_EDIT: Shift another 0.1 percentage point of ensemble weight from the EMA model to the live model while retaining the verified temperature.

EVIDENCE: Every increase from 50.0% through 50.5% live weight preserved 9,290 correct predictions and successively reduced cross-entropy; 51.0% reduced cross-entropy further but lost one correct prediction, motivating the next incremental boundary probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.505),
                        ema_ensemble + math.log(0.495),
=======
                        live_ensemble + math.log(0.506),
                        ema_ensemble + math.log(0.494),
>>>>>>> REPLACE