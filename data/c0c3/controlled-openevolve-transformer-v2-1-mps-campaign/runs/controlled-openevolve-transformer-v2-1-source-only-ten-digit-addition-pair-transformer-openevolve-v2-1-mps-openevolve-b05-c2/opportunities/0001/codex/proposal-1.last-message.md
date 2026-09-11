MECHANISM: One-neuron feed-forward bottleneck reduction

HYPOTHESIS: Reducing the sole MLP hidden width from 12 to 11 will lower parameters from 1644 to 1627 while retaining at least 99% accuracy, given the current model’s 99.96% accuracy margin.

INTENDED_EDIT: Change the default feed-forward width from 12 to 11 while preserving architecture, attention, training, and decoding behavior.

EVIDENCE: The current 1644-parameter design achieved 0.9996 accuracy at step 4999, motivating a conservative 17-parameter capacity reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE