MECHANISM: Bias-free terminal normalization with a time-safe training budget

HYPOTHESIS: Disabling the eight-parameter final LayerNorm bias will reduce the model to 1,376 parameters and retain at least 99% accuracy after 50,000 steps, while avoiding the timeout observed at 52,000 steps.

INTENDED_EDIT: Remove the final LayerNorm bias and shorten training from 52,000 to 50,000 steps.

EVIDENCE: The one-neuron model achieved 99.99% accuracy at 52,000 steps; the previous bias-free attempt produced no contrary accuracy evidence and failed only because verification timed out.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=50000)
>>>>>>> REPLACE