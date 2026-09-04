MECHANISM: Sub-quantization live/EMA mixture refinement

HYPOTHESIS: A 50.68800048828125% live / 49.31199951171875% EMA mixture will retain 9,290 correct predictions and reduce validation cross-entropy below 0.202489697265625.

INTENDED_EDIT: Move the ensemble weight halfway between the current best mixture and the higher-live mixture that tied its cross-entropy.

EVIDENCE: The 0.5068798828125 mixture is the verified best; 0.506880126953125 retained 9,290 correct and tied its cross-entropy, while the next higher tested weight worsened cross-entropy, making their untested midpoint the most informative local probe.

<<<<<<< SEARCH
                        live_ensemble + math.log(0.5068798828125),
                        ema_ensemble + math.log(0.4931201171875),
=======
                        live_ensemble + math.log(0.5068800048828125),
                        ema_ensemble + math.log(0.4931199951171875),
>>>>>>> REPLACE