MECHANISM: Eleventh one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 2 to 1 will lower the model from 1,068 to 1,053 learned parameters while retaining at least 99% accuracy, because all ten preceding one-neuron reductions met the requirement and the width-2 model achieved 99.78%.

INTENDED_EDIT: Change the default feed-forward width from 2 to 1 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 2 each removed 15 parameters while maintaining at least 99.78% accuracy; this uninterrupted trend makes the final single-neuron ablation the most informative capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=2)
=======
    p.add_argument("--d-ff", type=int, default=1)
>>>>>>> REPLACE