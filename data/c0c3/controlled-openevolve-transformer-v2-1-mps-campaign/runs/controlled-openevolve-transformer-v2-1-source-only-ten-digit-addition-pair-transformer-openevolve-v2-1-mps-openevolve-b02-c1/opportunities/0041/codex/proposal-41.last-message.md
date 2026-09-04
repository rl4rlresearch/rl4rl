MECHANISM: Fourth one-neuron feed-forward width ablation

HYPOTHESIS: Reducing the MLP width from 9 to 8 will lower the model from 1,173 to 1,158 learned parameters while retaining at least 99% accuracy, because the width-9 model achieved 99.98% and three consecutive one-neuron reductions preserved the required accuracy.

INTENDED_EDIT: Change the default feed-forward width from 9 to 8 while leaving attention, optimization, checkpointing, and protected generation unchanged.

EVIDENCE: Width reductions from 12 to 11, 11 to 10, and 10 to 9 each removed 15 parameters and achieved 99.91%, 99.96%, and 99.98% accuracy; unlike the failed shared-relative-bias experiment, this continues the strongest demonstrated parameter-reduction trend.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=9)
=======
    p.add_argument("--d-ff", type=int, default=8)
>>>>>>> REPLACE