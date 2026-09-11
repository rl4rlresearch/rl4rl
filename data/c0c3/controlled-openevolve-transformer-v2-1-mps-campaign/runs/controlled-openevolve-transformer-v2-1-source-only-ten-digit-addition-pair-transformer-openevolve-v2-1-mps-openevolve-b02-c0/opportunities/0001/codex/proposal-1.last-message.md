MECHANISM: Narrower feed-forward residual bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower parameters from 1,644 to 1,576 while retaining at least 99% accuracy, because the current model’s 99.96% accuracy provides margin for a modest MLP-width reduction.

INTENDED_EDIT: Change the default feed-forward width from 12 to 8 while preserving the attention architecture and training procedure.

EVIDENCE: The current 1,644-parameter design achieved 99.96% accuracy after 4,999 steps, motivating a conservative 68-parameter reduction in its MLP capacity.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE