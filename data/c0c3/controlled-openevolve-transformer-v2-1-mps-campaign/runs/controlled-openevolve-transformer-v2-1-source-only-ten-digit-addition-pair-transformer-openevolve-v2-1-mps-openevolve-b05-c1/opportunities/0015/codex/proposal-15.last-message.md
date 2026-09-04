MECHANISM: One-neuron MLP narrowing with extended convergence

HYPOTHESIS: Reducing `d_ff` from 12 to 11 will lower the model from 1,560 to 1,544 parameters and still reach at least 99% accuracy when trained for 12,000 steps.

INTENDED_EDIT: Use an 11-unit feed-forward layer and extend the convergence budget from 8,000 to 12,000 steps.

EVIDENCE: The 1,560-parameter quotient-space model improved from 97.76% at 5,000 steps to 99.87% at 8,000 steps, showing that additional training can overcome slow convergence; this motivates retesting the previously unsuccessful width-11 boundary with a larger budget.

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=12)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=8000)
=======
    p.add_argument("--d-ff", type=int, default=11)
    p.add_argument("--dropout", type=float, default=0.0)

    # optimization
    p.add_argument("--train-steps", type=int, default=12000)
>>>>>>> REPLACE