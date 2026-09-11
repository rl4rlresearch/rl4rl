MECHANISM: Lower-amplitude full-horizon label-smoothing anneal

HYPOTHESIS: Reducing initial smoothing from 0.04 to 0.03 while retaining the gradual zero endpoint will exceed 9,240 correct predictions by easing early underconfidence without prematurely removing regularization.

INTENDED_EDIT: Change only the cosine label-smoothing schedule from 0.04→0 to 0.03→0.

EVIDENCE: The 0.04→0 schedule achieved the best score, while 0.05→0 fell to 9,237 correct; the unsuccessful half-horizon anneal indicates smoothing should still decay across the full training run.

<<<<<<< SEARCH
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
=======
    smoothing = 0.015 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE