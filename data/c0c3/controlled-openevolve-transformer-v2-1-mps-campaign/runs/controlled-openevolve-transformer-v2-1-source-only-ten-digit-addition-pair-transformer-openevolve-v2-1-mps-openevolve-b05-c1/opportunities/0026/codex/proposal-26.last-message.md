MECHANISM: Single-neuron residual MLP with extended convergence

HYPOTHESIS: Reducing `d_ff` from 2 to 1 will lower the model from 1,400 to 1,384 parameters while retaining at least 99% accuracy when trained for 52,000 steps.

INTENDED_EDIT: Use a one-unit feed-forward layer and extend the default training budget from 48,000 to 52,000 steps.

EVIDENCE: Every successive `d_ff` reduction from 12 through 2 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=2` achieved 99.83% accuracy at 48,000 steps with 1,400 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=2)
=======
    p.add_argument("--d-ff", type=int, default=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=48000)
=======
    p.add_argument("--train-steps", type=int, default=52000)
>>>>>>> REPLACE