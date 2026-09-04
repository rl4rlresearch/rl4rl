MECHANISM: Time-safe bias-free terminal normalization

HYPOTHESIS: Removing the final LayerNorm bias will reduce the model from 1,384 to 1,376 parameters and retain at least 99% accuracy within a 48,000-step budget.

INTENDED_EDIT: Disable the eight-parameter final LayerNorm bias, train for 48,000 steps, and validate every 1,000 steps to limit verification time while preserving checkpoint selection.

EVIDENCE: The one-neuron model reached 99.99% at 52,000 steps, while the two-neuron model reached 99.83% at 48,000 steps; prior bias-free variants produced no contrary accuracy evidence and failed only by timing out at 50,000–52,000 steps.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=52000)
=======
    p.add_argument("--train-steps", type=int, default=48000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE