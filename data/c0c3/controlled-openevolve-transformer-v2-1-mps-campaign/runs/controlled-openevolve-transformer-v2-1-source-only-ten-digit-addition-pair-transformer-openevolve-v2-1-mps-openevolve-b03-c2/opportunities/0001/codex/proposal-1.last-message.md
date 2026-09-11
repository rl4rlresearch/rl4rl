MECHANISM: Narrower nonlinear feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while preserving at least 99% accuracy because the verified model has substantial accuracy headroom at 99.96% and retains its full attention and embedding capacity.

INTENDED_EDIT: Change the default feed-forward width from 12 to 8, removing 68 learned parameters.

EVIDENCE: The current `d_model=8, d_ff=12` design achieved 0.9996 accuracy with 1644 parameters, motivating a focused reduction in the feed-forward sublayer.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE