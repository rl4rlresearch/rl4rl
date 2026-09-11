MECHANISM: Upper-side sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68797302246094% live / 49.31202697753906% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the ensemble weight halfway from the current best mixture toward the nearest tested higher-live mixture.

EVIDENCE: The current 0.50687969970703125 live weight achieved the best verified cross-entropy; its equidistant tested neighbors were both worse, but the higher-live neighbor had lower cross-entropy than the lower-live neighbor, making the untested upper midpoint the most informative next probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068797302246094),
                        ema_ensemble + math.log(0.4931202697753906),
>>>>>>> REPLACE