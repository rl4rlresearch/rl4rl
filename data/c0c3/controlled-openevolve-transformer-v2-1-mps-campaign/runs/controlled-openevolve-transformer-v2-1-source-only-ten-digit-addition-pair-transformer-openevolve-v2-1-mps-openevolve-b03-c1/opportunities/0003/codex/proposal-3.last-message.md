MECHANISM: Single-neuron feed-forward narrowing

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower parameters from 1,644 to 1,627 while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one feed-forward hidden unit while preserving the attention architecture and training procedure.

EVIDENCE: The `d_ff=12` baseline reached 99.96%, while the larger reduction to `d_ff=8` fell to 71%; testing the smallest possible width reduction is the most informative next capacity-boundary experiment.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE