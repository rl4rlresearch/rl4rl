MECHANISM: Narrower mean-preserving ensemble-loss curriculum

HYPOTHESIS: A 0.65–0.85 curriculum will exceed 9,237 correct predictions by retaining beneficial stage-dependent supervision while avoiding the excessive specialization observed with the wider 0.55–0.95 range.

INTENDED_EDIT: Narrow the linear ensemble-loss curriculum from 0.60–0.90 to 0.65–0.85 while preserving its mean weight of 0.75.

EVIDENCE: Static 0.75 reached 9,236 correct, the 0.60–0.90 curriculum improved to 9,237, and widening it to 0.55–0.95 regressed to 9,235; this motivates testing an intermediate curriculum amplitude.

<<<<<<< SEARCH
    ensemble_weight = 0.60 + 0.30 * progress
=======
    ensemble_weight = 0.65 + 0.20 * progress
>>>>>>> REPLACE