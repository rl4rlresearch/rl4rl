MECHANISM: One-unit MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 11 to 10 will lower the model from 881 to 866 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one hidden unit from the transformer MLP without changing the proven rank-four lexical representation or attention routing.

EVIDENCE: The rank-four model achieved 99.98% accuracy, while reducing lexical rank collapsed accuracy to 79.28%; testing a single-unit MLP reduction is therefore the smallest informative compression along a different capacity axis.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE