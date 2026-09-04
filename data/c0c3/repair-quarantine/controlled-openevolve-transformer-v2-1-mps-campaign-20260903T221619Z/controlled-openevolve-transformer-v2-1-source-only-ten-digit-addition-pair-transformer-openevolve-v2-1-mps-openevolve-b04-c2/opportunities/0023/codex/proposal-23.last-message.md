MECHANISM: Pre-MLP LayerNorm bias absorption

HYPOTHESIS: Combining the verified value-bias removal with elimination of the eight-dimensional pre-MLP LayerNorm bias will yield 1,613 parameters and at least 99% accuracy, because that LayerNorm shift is exactly absorbable by the retained `fc1` bias.

INTENDED_EDIT: Retain only the learned attention query bias, then remove `ln2.bias` while preserving module initialization and all downstream learned weights and biases.

EVIDENCE: The 1,621-parameter query-only attention-bias design achieved 99.98% accuracy, demonstrating substantial margin after an exact bias absorption; `ln2.bias` has the same direct redundancy since `fc1(W)` maps its position-independent shift into the unrestricted learned `fc1.bias`.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the original parameter slot and initialization RNG consumption,
        # but store only the query and value biases. A shared key bias cancels
        # from every row of the attention softmax.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)
        # Any learned LayerNorm shift enters the MLP only through fc1 and is
        # exactly absorbable by its unrestricted bias.
        self.ln2.bias = None
>>>>>>> REPLACE