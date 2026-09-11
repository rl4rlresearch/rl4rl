MECHANISM: Widened ensemble-supervision curriculum

HYPOTHESIS: Expanding the ensemble-loss allocation from 55% early to 95% late will exceed 9,247 correct predictions by emphasizing individual-view feature learning early and the validation-aligned crop/flip ensemble near convergence.

INTENDED_EDIT: Increase the cosine ensemble-weight amplitude from ±0.15 to ±0.20 while preserving its 0.75 mean and all other settings.

EVIDENCE: The analogous 25%–50% pair-loss curriculum produced the current 9,247-correct best result, while direct flip-consistency supervision regressed to 9,241; this tests stronger temporal specialization using the already-successful supervised ensemble objective.

<<<<<<< SEARCH
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
=======
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
>>>>>>> REPLACE