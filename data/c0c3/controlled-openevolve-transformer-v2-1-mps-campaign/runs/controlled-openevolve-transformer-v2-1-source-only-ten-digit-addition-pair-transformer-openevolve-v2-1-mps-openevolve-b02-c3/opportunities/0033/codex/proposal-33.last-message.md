MECHANISM: Pre-attention constant-shift absorption

HYPOTHESIS: Removing the five pre-attention LayerNorm bias coefficients will produce a 1,044-parameter transformer with at least 99% accuracy, because their query effect is absorbable by `q_bias`, key shifts cancel in softmax, and value shifts are absorbable by the attention output bias.

INTENDED_EDIT: Replace the five-direction pre-attention LayerNorm with an affine-free LayerNorm while preserving the qualified factorized attention architecture.

EVIDENCE: The current 1,049-parameter model achieved 99.93% accuracy, and prior pre-attention gauge evidence established the exact query-bias, key-softmax, and attention-output-bias absorptions used by this reduction.

<<<<<<< SEARCH
        self.ln1 = GaugeFixedLayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE