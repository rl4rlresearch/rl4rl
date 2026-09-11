MECHANISM: Remove affine redundancy before the MLP

HYPOTHESIS: Disabling the second LayerNorm bias will reduce parameters from 1,636 to 1,628 while retaining at least 99% accuracy, because the following `fc1` bias can represent the same pre-GELU offset.

INTENDED_EDIT: Construct `ln2` without a learned bias while preserving its learned scale, the full `d_ff=12` capacity, initialization RNG stream, and training procedure.

EVIDENCE: The 1,636-parameter RNG-aligned design achieved 99.09%, while reducing feed-forward width to 11 or 8 failed badly; this motivates removing an exactly absorbable affine parameter instead of reducing MLP capacity.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
>>>>>>> REPLACE