MECHANISM: Cosine-decayed label smoothing with a 0.008 floor

HYPOTHESIS: Ending smoothing at 0.008 will exceed 9,258 correct predictions by adding modest late boundary sharpening while retaining substantially more regularization than the zero-floor schedule that regressed to 9,254.

INTENDED_EDIT: Lower the cosine schedule’s final smoothing from 0.01 to 0.008 while preserving its initial value of 0.02.

EVIDENCE: The 0.01 floor improved to 9,258 correct and beat the 0.013 floor on cross-entropy, whereas decay to zero lost four correct predictions; 0.008 probes the promising interval immediately below the current best without approaching complete smoothing removal.

<<<<<<< SEARCH
    smoothing = 0.01 + 0.005 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.008 + 0.006 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE