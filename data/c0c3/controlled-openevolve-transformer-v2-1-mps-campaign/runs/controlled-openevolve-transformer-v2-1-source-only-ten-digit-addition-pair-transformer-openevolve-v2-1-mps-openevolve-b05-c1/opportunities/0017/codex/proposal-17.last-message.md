MECHANISM: Progressive MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 10 to 9 will lower the model from 1,528 to 1,512 parameters while retaining at least 99% accuracy when trained for 20,000 steps.

INTENDED_EDIT: Use a 9-unit feed-forward layer and extend the default training budget from 16,000 to 20,000 steps.

EVIDENCE: Successive reductions from `d_ff=12` to 11 and then 10 achieved 99.87%, 99.93%, and 99.98% with progressively longer training, supporting the next one-neuron reduction with another 4,000-step allowance.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=10)
=======
    p.add_argument("--d-ff", type=int, default=9)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=16000)
=======
    p.add_argument("--train-steps", type=int, default=20000)
>>>>>>> REPLACE