MECHANISM: Partial train–test translation-kernel alignment

HYPOTHESIS: Halfway interpolation toward the inference translation kernel will retain or exceed 9,283 correct predictions while improving cross-entropy relative to the beta2=0.96 baseline.

INTENDED_EDIT: Restore AdamW beta2=0.96 and replace symmetric training translations with a 50% interpolation toward the vertical-biased, distance-stratified TTA weights.

EVIDENCE: Full kernel alignment reached 9,281 correct but lowered cross-entropy from 0.2093681 to 0.2084849; partial alignment tests whether that confidence gain can be captured without losing the baseline’s two additional correct predictions.

<<<<<<< SEARCH
        betas=(0.9, 0.95),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.09000, 0.36000, 0.09000, 0.00625,
            0.00625, 0.04500, 0.09000, 0.04500, 0.00625,
            0.00625, 0.00625, 0.00625, 0.00625, 0.00625,
        )
=======
        (
            0.0046875, 0.0062500, 0.0078125, 0.0062500, 0.0046875,
            0.0062500, 0.0450000, 0.0950000, 0.0450000, 0.0062500,
            0.0078125, 0.0850000, 0.3600000, 0.0850000, 0.0078125,
            0.0062500, 0.0450000, 0.0950000, 0.0450000, 0.0062500,
            0.0046875, 0.0062500, 0.0078125, 0.0062500, 0.0046875,
        )
>>>>>>> REPLACE