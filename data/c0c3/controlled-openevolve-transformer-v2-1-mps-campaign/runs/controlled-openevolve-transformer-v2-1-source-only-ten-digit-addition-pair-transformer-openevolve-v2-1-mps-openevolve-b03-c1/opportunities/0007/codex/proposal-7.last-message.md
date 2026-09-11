MECHANISM: Remove attention-input affine redundancy

HYPOTHESIS: Disabling the first LayerNorm bias will reduce parameters from 1,628 to 1,620 while retaining at least 99% accuracy, because its query and value offsets can be represented by the existing Q/V biases while its key offset is softmax-invariant.

INTENDED_EDIT: Construct `ln1` without a learned bias while preserving its learned scale, attention dimensions, Q/V biases, feed-forward capacity, initialization, and training procedure.

EVIDENCE: Removing the analogous `ln2` bias improved the available design to 1,628 parameters with 99.77% accuracy; unlike the failed value-bias removal, this edit preserves the attention biases that appear important for optimization.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE