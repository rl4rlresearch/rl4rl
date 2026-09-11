MECHANISM: Narrower feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 10 will lower the model from 1,644 to 1,610 learned parameters while retaining at least 99% accuracy after fresh training.

INTENDED_EDIT: Reduce the default MLP hidden width by two units without changing attention, embeddings, decoding, or training.

EVIDENCE: The current 1,644-parameter model achieved 99.96% accuracy, providing accuracy headroom for a conservative 34-parameter MLP reduction.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE