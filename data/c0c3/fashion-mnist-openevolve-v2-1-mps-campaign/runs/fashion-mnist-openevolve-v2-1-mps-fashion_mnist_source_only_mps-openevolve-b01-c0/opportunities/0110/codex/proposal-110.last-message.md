MECHANISM: Lower-side sub-quantization live/EMA mixture refinement retry

HYPOTHESIS: A 50.6879638671875% live / 49.3120361328125% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Retry the unresolved lower-side midpoint by decreasing the live-model ensemble weight from 0.5068798828125 to 0.506879638671875 and increasing the EMA complement.

EVIDENCE: The current mixture is best at 9,290 correct and 0.202489697265625 cross-entropy; the only verification of this adjacent midpoint timed out, so retrying it resolves the most informative remaining local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.506879638671875),
                        ema_ensemble + math.log(0.493120361328125),
>>>>>>> REPLACE