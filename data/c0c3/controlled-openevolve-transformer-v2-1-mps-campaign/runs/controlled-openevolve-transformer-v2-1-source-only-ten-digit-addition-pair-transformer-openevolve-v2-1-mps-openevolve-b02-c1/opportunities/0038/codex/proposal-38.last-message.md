MECHANISM: Second one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 11 to 10 will lower the model from 1,203 to 1,188 learned parameters while retaining at least 99% accuracy, because the width-11 model achieved 99.91% without changing the attention or decoding pathways.

INTENDED_EDIT: Change the default feed-forward width from 11 to 10; leave training, optimizer behavior, checkpointing, and protected generation unchanged.

EVIDENCE: The immediately preceding width reduction from 12 to 11 removed 15 parameters and still achieved 99.91% accuracy, making the next single-neuron ablation the most direct capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=11)
=======
    p.add_argument("--d-ff", type=int, default=10)
>>>>>>> REPLACE