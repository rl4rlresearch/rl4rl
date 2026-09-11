MECHANISM: Single-neuron feed-forward width ablation

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower the model from 1,644 to 1,627 deduplicated learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one MLP hidden unit while preserving every attention, MLP, and normalization bias.

EVIDENCE: The 1,644-parameter design achieved 99.96% accuracy, while broader and attention-specific bias removal fell below the threshold; this motivates a conservative capacity reduction that leaves all apparently important biases intact.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE