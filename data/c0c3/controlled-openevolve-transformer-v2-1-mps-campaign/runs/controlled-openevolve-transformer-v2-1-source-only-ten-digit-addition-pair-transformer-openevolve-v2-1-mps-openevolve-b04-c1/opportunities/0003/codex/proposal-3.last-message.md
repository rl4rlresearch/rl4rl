MECHANISM: Incremental feed-forward bottleneck narrowing

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower parameters from 1644 to 1627 while retaining at least 99% accuracy because it preserves the verified 8-wide residual stream, all projection biases, and the full two-head attention module.

INTENDED_EDIT: Change the default MLP hidden width from 12 to 11 without altering the architecture or training schedule otherwise.

EVIDENCE: The 8-wide, bias-enabled baseline reached 99.96% accuracy, while narrowing the residual stream to 6 and removing all linear biases both failed; this motivates a conservative one-unit reduction confined to the MLP.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE