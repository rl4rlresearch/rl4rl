MECHANISM: Stronger mean-preserving ensemble-loss curriculum

HYPOTHESIS: Increasing the curriculum range from 0.60–0.90 to 0.55–0.95 will exceed 9,237 correct predictions by strengthening early per-view feature learning and late deployed-ensemble alignment while preserving the successful average ensemble weight of 0.75.

INTENDED_EDIT: Steepen the linear ensemble-supervision curriculum without changing its mean, architecture, views, optimizer, or evaluation.

EVIDENCE: Static 0.75 produced 9,236 correct, while the mean-preserving 0.60–0.90 curriculum improved this to 9,237; widening that curriculum directly tests whether greater stage specialization extends the observed gain.

<<<<<<< SEARCH
    ensemble_weight = 0.60 + 0.30 * progress
=======
    ensemble_weight = 0.55 + 0.40 * progress
>>>>>>> REPLACE