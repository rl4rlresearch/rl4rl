MECHANISM: Bias-free terminal normalization with sparse validation

HYPOTHESIS: Removing the final LayerNorm bias reduces the model from 1,384 to 1,376 parameters, while reducing validation frequency allows the proven 52,000-step budget to finish and retain at least 99% accuracy.

INTENDED_EDIT: Disable the eight-parameter final LayerNorm bias and evaluate every 1,000 steps instead of every 300 steps.

EVIDENCE: The 1,384-parameter model achieved 99.99% accuracy at 52,000 steps; both prior bias-free attempts timed out without contrary accuracy evidence, including at 50,000 steps, motivating reduced evaluation overhead rather than another training-budget cut.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE