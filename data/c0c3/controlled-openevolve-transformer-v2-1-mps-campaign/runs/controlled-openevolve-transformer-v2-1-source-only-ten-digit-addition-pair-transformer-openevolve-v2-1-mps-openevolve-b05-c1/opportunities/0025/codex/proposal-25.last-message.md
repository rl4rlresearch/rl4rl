MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 3 to 2 will lower the model from 1,416 to 1,400 parameters while retaining at least 99% accuracy when trained for 48,000 steps.

INTENDED_EDIT: Use a two-unit feed-forward layer and extend the default training budget from 44,000 to 48,000 steps.

EVIDENCE: Every successive `d_ff` reduction from 12 through 3 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=3` achieved 99.98% accuracy at 44,000 steps with 1,416 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=3)
=======
    p.add_argument("--d-ff", type=int, default=2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=44000)
=======
    p.add_argument("--train-steps", type=int, default=48000)
>>>>>>> REPLACE