MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 4 to 3 will lower the model from 1,432 to 1,416 parameters while retaining at least 99% accuracy when trained for 44,000 steps.

INTENDED_EDIT: Use a 3-unit feed-forward layer and extend the default training budget from 40,000 to 44,000 steps.

EVIDENCE: Every successive `d_ff` reduction from 12 through 4 met the accuracy requirement with 4,000 additional training steps per removed neuron; most recently, `d_ff=4` achieved 99.97% accuracy at 40,000 steps with 1,432 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=4)
=======
    p.add_argument("--d-ff", type=int, default=3)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=40000)
=======
    p.add_argument("--train-steps", type=int, default=44000)
>>>>>>> REPLACE