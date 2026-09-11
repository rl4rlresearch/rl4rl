MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 8 to 7 will lower the model from 1,496 to 1,480 parameters while retaining at least 99% accuracy when trained for 28,000 steps.

INTENDED_EDIT: Use a 7-unit feed-forward layer and extend the default training budget from 24,000 to 28,000 steps.

EVIDENCE: Successive `d_ff` reductions from 12 through 8 all exceeded 99% accuracy when paired with 4,000 additional training steps per removed neuron; most recently, `d_ff=8` achieved 99.99% at 24,000 steps with 1,496 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=8)
=======
    p.add_argument("--d-ff", type=int, default=7)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=24000)
=======
    p.add_argument("--train-steps", type=int, default=28000)
>>>>>>> REPLACE