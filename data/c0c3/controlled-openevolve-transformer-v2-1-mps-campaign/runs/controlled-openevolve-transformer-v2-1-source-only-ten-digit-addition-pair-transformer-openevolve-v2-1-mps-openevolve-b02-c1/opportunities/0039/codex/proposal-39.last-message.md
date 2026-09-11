MECHANISM: Third one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 10 to 9 will lower the model from 1,188 to 1,173 learned parameters while retaining at least 99% accuracy, because the width-10 model achieved 99.96% without changing the attention or decoding pathways.

INTENDED_EDIT: Change the default feed-forward width from 10 to 9; leave training, optimizer behavior, checkpointing, and protected generation unchanged.

EVIDENCE: Consecutive reductions from width 12 to 11 and then 10 each removed 15 parameters while achieving 99.91% and 99.96% accuracy, making the next single-neuron ablation the clearest capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=10)
=======
    p.add_argument("--d-ff", type=int, default=9)
>>>>>>> REPLACE