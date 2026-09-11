MECHANISM: One-neuron MLP capacity titration

HYPOTHESIS: Reducing the MLP width from 12 to 11 will lower parameters from 1636 to 1619 while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one hidden unit from the sole feed-forward layer while preserving the successful attention and normalization design.

EVIDENCE: The 12-unit model achieved 99.84%, while the larger reduction to 8 units collapsed to 44.84%; testing 11 isolates the smallest capacity reduction and avoids altering optimization-sensitive attention biases.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE