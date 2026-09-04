MECHANISM: Lower-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68797607421875% live / 49.31202392578125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Move the ensemble weight halfway between the current best mixture and the lower-live mixture that retained accuracy but had slightly worse cross-entropy.

EVIDENCE: The current 0.5068798828125 live weight is best at 9,290 correct and 0.202489697265625 cross-entropy; 0.506879638671875 retained 9,290 correct but produced 0.2024896987915039, making their untested midpoint the most informative remaining local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.5068797607421875),
                        ema_ensemble + math.log(0.4931202392578125),
>>>>>>> REPLACE