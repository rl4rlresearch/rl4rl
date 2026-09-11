MECHANISM: Intermediate feed-forward bottleneck search

HYPOTHESIS: A `d_ff` width of 10 will retain at least 99% accuracy while reducing the model from 1644 to 1610 parameters, locating whether the capacity threshold lies between the failed width 8 and verified width 12.

INTENDED_EDIT: Change the default feed-forward width from 12 to 10.

EVIDENCE: The width-12 model reached 99.96% accuracy, while width 8 fell to 71.29%; testing the midpoint is the most informative next capacity reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE