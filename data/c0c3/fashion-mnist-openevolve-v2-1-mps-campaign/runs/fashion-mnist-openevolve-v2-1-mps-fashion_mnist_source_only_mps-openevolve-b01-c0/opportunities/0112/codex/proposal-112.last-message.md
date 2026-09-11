MECHANISM: Upper-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.687982177734375% live / 49.312017822265625% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969650268556.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest tested higher-live mixture.

EVIDENCE: The current 0.5068797607421875 live weight achieved the best cross-entropy; the equidistant tested neighbors were both worse, but the higher-live neighbor was closer in cross-entropy than the lower-live neighbor, making the upper midpoint the most informative remaining probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068797607421875),
                        ema_ensemble + math.log(0.4931202392578125),
=======
                        live_ensemble + math.log(0.50687982177734375),
                        ema_ensemble + math.log(0.49312017822265625),
>>>>>>> REPLACE