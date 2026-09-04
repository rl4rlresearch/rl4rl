MECHANISM: Narrower feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower deduplicated parameters from 1,644 to 1,576 while retaining at least 99% accuracy after 5,000 training steps.

INTENDED_EDIT: Change the default feed-forward width to 8 without altering the learned attention architecture or training procedure.

EVIDENCE: The current 1,644-parameter model achieved 99.96% accuracy at step 4,999, providing substantial accuracy margin for a moderate 68-parameter reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE