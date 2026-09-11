MECHANISM: Pre-attention LayerNorm bias elimination

HYPOTHESIS: Removing `ln1`’s eight zero-initialized biases will reduce the qualified model to 1,470 parameters while retaining at least 99% accuracy, because query and value biases preserve the same function class and key offsets are softmax-invariant.

INTENDED_EDIT: Make the pre-attention LayerNorm bias-free while retaining its learned scales and all attention biases.

EVIDENCE: The 1,478-parameter model achieved 99.88%; unlike the timed-out `fc1` quotient attempts, this exact eight-parameter redundancy requires no reconstructed weights or extra per-step optimizer work. Prior query-bias failures motivate retaining the dedicated query bias rather than removing attention bias capacity.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE