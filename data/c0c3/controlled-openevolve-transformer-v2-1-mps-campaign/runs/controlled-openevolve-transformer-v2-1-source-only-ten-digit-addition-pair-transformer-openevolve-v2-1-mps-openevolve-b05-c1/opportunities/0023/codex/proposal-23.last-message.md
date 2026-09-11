MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 5 to 4 will lower the model from 1,448 to 1,432 parameters while retaining at least 99% accuracy when trained for 40,000 steps.

INTENDED_EDIT: Use a 4-unit feed-forward layer and extend the default training budget from 36,000 to 40,000 steps.

EVIDENCE: Every successive `d_ff` reduction from 12 through 5 met the accuracy requirement with 4,000 additional steps per removed neuron; most recently, `d_ff=5` achieved 99.96% accuracy at 36,000 steps with 1,448 parameters.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=5)
=======
    p.add_argument("--d-ff", type=int, default=4)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=36000)
=======
    p.add_argument("--train-steps", type=int, default=40000)
>>>>>>> REPLACE