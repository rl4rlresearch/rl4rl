MECHANISM: Fifth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 8 to 7 will lower the model from 1,158 to 1,143 learned parameters while retaining at least 99% accuracy, because four consecutive one-neuron reductions preserved at least 99.91% accuracy and the width-8 model achieved 99.99%.

INTENDED_EDIT: Change the default feed-forward width from 8 to 7 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 through 8 each removed 15 parameters and achieved 99.91%, 99.96%, 99.98%, and 99.99% accuracy; this uninterrupted trend makes the next one-neuron ablation the most informative capacity-boundary test.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE