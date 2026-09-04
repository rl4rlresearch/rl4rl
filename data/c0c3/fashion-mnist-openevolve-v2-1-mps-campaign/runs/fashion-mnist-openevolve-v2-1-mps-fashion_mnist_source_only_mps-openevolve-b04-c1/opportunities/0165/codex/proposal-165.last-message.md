MECHANISM: Higher-update stochastic optimization

HYPOTHESIS: Reducing batch size from 64 to 56 will exceed 9,328 correct predictions by providing roughly 14% more optimizer updates and mildly noisier gradients within the fixed 100,000-example exposure.

INTENDED_EDIT: Use 56-example minibatches while preserving the proven architecture, learning-rate schedule, loss annealing, and parameter EMA.

EVIDENCE: Lowering the terminal learning-rate multiplier reduced correctness to 9,302, while raising it reduced correctness to 9,309, suggesting the current per-step schedule is well tuned; changing batch size tests additional optimization opportunities without perturbing that schedule or revisiting the tightly bracketed EMA settings.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 1.0
=======
BATCH_SIZE = 56
GRAD_CLIP_NORM = 1.0
>>>>>>> REPLACE