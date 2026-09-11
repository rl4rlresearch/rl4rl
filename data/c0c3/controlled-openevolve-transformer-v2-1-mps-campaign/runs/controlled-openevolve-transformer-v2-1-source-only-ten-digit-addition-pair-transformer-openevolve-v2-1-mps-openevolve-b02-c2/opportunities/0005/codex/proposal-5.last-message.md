MECHANISM: Single-neuron MLP bottleneck reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower the model from 1,644 to 1,627 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Remove one feed-forward hidden unit while preserving attention, embeddings, initialization, decoding, and training settings.

EVIDENCE: The 1,644-parameter `d_ff=12` model achieved 99.96%, while `d_ff=10` collapsed to 9.07%; testing the intermediate width isolates the minimum viable MLP capacity.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE