MECHANISM: Narrower feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower parameters from 1,644 to 1,576 while retaining at least 99% accuracy.

INTENDED_EDIT: Change the default MLP hidden width from 12 to 8 without altering attention or training.

EVIDENCE: The current 1,644-parameter model achieves 99.96% accuracy, leaving substantial margin above the 99% requirement for an isolated 68-parameter reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE