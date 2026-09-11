MECHANISM: Single-neuron MLP width reduction

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will preserve at least 99% accuracy while removing 17 learned parameters, because it retains the full positional embeddings and all projection biases shown to be important by the two failed compressions.

INTENDED_EDIT: Change the default feed-forward width from 12 to 11 without altering the successful attention, embedding, bias, or training configuration.

EVIDENCE: The 1,644-parameter baseline reached 99.96% accuracy, while positional compression and wholesale bias removal collapsed accuracy; this motivates a smaller, orthogonal reduction in MLP capacity.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
=======
    p.add_argument("--d-ff", type=int, default=11)
>>>>>>> REPLACE