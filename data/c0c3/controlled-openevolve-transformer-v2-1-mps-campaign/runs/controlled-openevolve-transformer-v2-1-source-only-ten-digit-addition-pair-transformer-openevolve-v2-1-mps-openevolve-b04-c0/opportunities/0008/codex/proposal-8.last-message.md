MECHANISM: Pre-MLP bias reparameterization

HYPOTHESIS: Removing `ln2.bias` will reduce parameters from 1628 to 1620 while retaining at least 99% accuracy, because its contribution is exactly absorbable by the independent `fc1.bias` before GELU.

INTENDED_EDIT: Disable only the second LayerNorm’s bias while preserving its learned scale and all MLP parameters.

EVIDENCE: Tying `ln2.bias` to the post-nonlinearity `fc2.bias` failed at 11.24%; this tests the mathematically valid redundancy with the pre-nonlinearity `fc1.bias` without narrowing the MLP.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE