MECHANISM: Cosine-shaped ensemble-supervision curriculum

HYPOTHESIS: A cosine transition from 0.60 to 0.90 will exceed 9,237 correct predictions by strengthening early per-view learning and late ensemble alignment without exceeding the empirically best curriculum endpoints or changing the successful mean weight of 0.75.

INTENDED_EDIT: Replace the linear ensemble-loss curriculum with a smooth cosine curriculum having the same endpoints and mean.

EVIDENCE: The 0.60–0.90 linear curriculum improved the static 0.75 result from 9,236 to 9,237 correct, while both narrower and wider ranges regressed; this isolates temporal concentration while preserving the supported amplitude.

<<<<<<< SEARCH
    ensemble_weight = 0.60 + 0.30 * progress
=======
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
>>>>>>> REPLACE