MECHANISM: Front-loaded ensemble-objective curriculum

HYPOTHESIS: Raising the initial ensemble-loss weight from 0.5 to 0.75 will exceed 9,330 correct predictions by aligning more of the fixed training budget with the flip-ensemble used for validation, without increasing runtime.

INTENDED_EDIT: Change the ensemble-loss curriculum from 0.5→1.0 to 0.75→1.0 while retaining individual-view supervision early in training.

EVIDENCE: The best verified design reaches 9,330 correct as ensemble supervision becomes dominant; this conservatively tests earlier validation-objective alignment, while recent more complex loss changes provided no completed contrary result.

<<<<<<< SEARCH
    ensemble_weight = 0.5 + 0.5 * progress
=======
    ensemble_weight = 0.75 + 0.25 * progress
>>>>>>> REPLACE