MECHANISM: One-unit feed-forward bottleneck reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will remove 17 learned parameters while retaining at least 99% accuracy after 5,000 training steps.

INTENDED_EDIT: Change the default transformer feed-forward width to 11.

EVIDENCE: The 12-wide model achieved 99.96% accuracy, while width 8 failed at 45.79%; testing the nearest smaller width conservatively probes the capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE