MECHANISM: Narrower nonlinear feed-forward bottleneck

HYPOTHESIS: Reducing `d_ff` from 12 to 8 will lower parameters from 1644 to 1576 while retaining at least 99% accuracy because the current model reaches 99.96%, leaving substantial accuracy margin.

INTENDED_EDIT: Reduce the single transformer block’s default feed-forward width by one third while leaving attention, embeddings, training, and decoding unchanged.

EVIDENCE: The verified `d_model=8, d_ff=12` baseline achieves 0.9996 accuracy with 1644 parameters; a controlled feed-forward-width reduction is the most direct low-risk capacity ablation.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE