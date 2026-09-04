MECHANISM: Upper-side sixteenth-step live/EMA mixture refinement

HYPOTHESIS: A 50.68797016143799% live / 49.31202983856201% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Move the live-model ensemble weight halfway from the current best toward the nearest tested higher-live mixture that tied its cross-entropy.

EVIDENCE: The current 0.50687969970703125 weight and the higher 0.5068797035217285 weight both achieved 9,290 correct with 0.20248969497680663 cross-entropy, while farther upper and lower probes were worse; their untested midpoint is the most informative remaining local refinement.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.50687969970703125),
                        ema_ensemble + math.log(0.49312030029296875),
=======
                        live_ensemble + math.log(0.5068797016143799),
                        ema_ensemble + math.log(0.4931202983856201),
>>>>>>> REPLACE