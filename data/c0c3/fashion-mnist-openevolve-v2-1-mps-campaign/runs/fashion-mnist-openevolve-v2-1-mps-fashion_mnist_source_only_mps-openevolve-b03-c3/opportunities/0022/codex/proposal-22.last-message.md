MECHANISM: Intermediate-width progressive flip-orbit supervision

HYPOTHESIS: A cosine curriculum from 55% to 95% flip-ensemble weight will exceed 9,237 correct predictions by retaining useful endpoint diversity while avoiding the weaker extremes of the 50%–100% schedule.

INTENDED_EDIT: Narrow the current ensemble-loss curriculum from 50%–100% to 55%–95%, preserving its 75% average and progression direction.

EVIDENCE: The 60%–90% curriculum achieved 9,237 correct versus 9,235 for 50%–100%, while narrowing further to 65%–85% fell to 9,229; testing the unmeasured intermediate width directly brackets the strongest region.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
=======
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
>>>>>>> REPLACE