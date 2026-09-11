MECHANISM: Lower-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.6879638671875% live / 49.3120361328125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Move the ensemble weight halfway between the current best mixture and the tested lower-live midpoint.

EVIDENCE: The current 0.5068798828125 live weight has the best verified cross-entropy; 0.50687939453125 retained 9,290 correct but was worse, while the corresponding upper-side refinement was also worse, making the untested lower half-midpoint the most informative remaining local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.506879638671875),
                        ema_ensemble + math.log(0.493120361328125),
>>>>>>> REPLACE