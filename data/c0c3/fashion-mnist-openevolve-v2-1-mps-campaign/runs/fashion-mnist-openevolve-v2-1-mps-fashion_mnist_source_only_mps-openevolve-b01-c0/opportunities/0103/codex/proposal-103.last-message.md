MECHANISM: Local live/EMA cross-entropy refinement

HYPOTHESIS: A 50.687939453125% live / 49.312060546875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Move the ensemble weight halfway from the current best toward the nearby lower-live result.

EVIDENCE: The current 0.5068798828125 live weight achieved the best cross-entropy, while both adjacent tested weights were worse; the lower-live neighbor was closer in cross-entropy than the upper-live neighbor, motivating a lower-side midpoint probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.50687939453125),
                        ema_ensemble + math.log(0.49312060546875),
>>>>>>> REPLACE