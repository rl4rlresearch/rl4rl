MECHANISM: Narrower learned feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy.

INTENDED_EDIT: Reduce the single transformer block’s default MLP width by one third.

EVIDENCE: The current 1644-parameter model reaches 99.96% accuracy, providing substantial accuracy margin for a conservative 68-parameter reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE