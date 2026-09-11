MECHANISM: Attention pre-normalization affine absorption and zero-mean coordinate elimination

HYPOTHESIS: Making the attention-preceding LayerNorm non-affine and omitting its linearly dependent final normalized coordinate from QKV projection will reduce the model from 1,600 to 1,560 parameters while preserving at least 99% accuracy.

INTENDED_EDIT: Remove the first LayerNorm’s 16 affine parameters and reduce the QKV projection from eight to seven input coordinates, retaining the learned causal attention computation and residual width.

EVIDENCE: The analogous non-affine MLP LayerNorm retained 99.92% accuracy, and eliminating its dependent normalized coordinate then reached 99.99% at 1,600 parameters; this motivates applying the same exact zero-mean reparameterization to the attention input.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = nn.Linear(d_model - 1, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
=======
        qkv = self.qkv(x[..., :-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE