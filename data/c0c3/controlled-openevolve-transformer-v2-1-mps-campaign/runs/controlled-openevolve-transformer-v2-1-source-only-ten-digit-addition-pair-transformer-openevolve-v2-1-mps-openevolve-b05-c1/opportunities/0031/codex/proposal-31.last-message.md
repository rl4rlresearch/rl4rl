MECHANISM: Bias-free terminal normalization

HYPOTHESIS: Removing only the final LayerNorm bias will reduce the model from 1,384 to 1,376 parameters while retaining at least 99% accuracy within 52,000 steps.

INTENDED_EDIT: Disable the eight-parameter bias in the final LayerNorm while preserving its learned scale and all attention and MLP parameters.

EVIDENCE: The current model achieved 99.99% accuracy, whereas removing the one-neuron MLP collapsed accuracy to 73.76%; this motivates preserving the computational core and testing a zero-overhead normalization parameter reduction that does not risk the runtime failures of quotient reconstruction.

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE