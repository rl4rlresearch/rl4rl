MECHANISM: Narrow feed-forward bottleneck

HYPOTHESIS: Reducing the MLP width from 12 to 8 will remove 68 learned parameters while retaining at least 99% accuracy after 5,000 training steps.

INTENDED_EDIT: Change the default transformer feed-forward width to 8.

EVIDENCE: The current 1,644-parameter model achieves 99.96% accuracy, providing enough margin to test a moderate reduction in MLP capacity while preserving the attention architecture.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE