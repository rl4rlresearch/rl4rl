MECHANISM: Single-unit feed-forward bottleneck reduction

HYPOTHESIS: A `d_ff` width of 11 will retain at least 99% accuracy while reducing parameters from 1,644 to 1,627.

INTENDED_EDIT: Change the default MLP hidden width from 12 to 11 while preserving attention, biases, and training settings.

EVIDENCE: Width 12 achieved 99.96% accuracy, while widths 10 and 8 failed; width 11 is the most informative remaining capacity boundary.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE