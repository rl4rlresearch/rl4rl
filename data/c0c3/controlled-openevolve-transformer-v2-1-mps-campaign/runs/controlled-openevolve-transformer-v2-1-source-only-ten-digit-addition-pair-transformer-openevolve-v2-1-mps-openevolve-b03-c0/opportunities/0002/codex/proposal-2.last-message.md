MECHANISM: Intermediate feed-forward bottleneck width

HYPOTHESIS: A `d_ff` width of 10 will retain at least 99% accuracy while reducing learned parameters from 1,644 to 1,610.

INTENDED_EDIT: Change the default MLP hidden width from 12 to 10 while leaving attention and training unchanged.

EVIDENCE: Width 12 achieved 99.96% accuracy, while width 8 fell to 93.9%; testing the midpoint isolates whether the capacity threshold lies above or below width 10.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE