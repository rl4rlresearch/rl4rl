MECHANISM: Third pre-attention LayerNorm-bias absorption gauge

HYPOTHESIS: Fixing a third `ln1.bias` coordinate will reduce the verified 1,605-parameter model to 1,604 parameters while retaining at least 99% accuracy, because the first two removals achieved 99.96% and 99.54% and the learned attention projections can absorb another constant LayerNorm offset.

INTENDED_EDIT: Retain five rather than six learned `ln1.bias` coordinates by reconstructing its final three coordinates as zeros.

EVIDENCE: Fixing the second `ln1.bias` coordinate produced the current 1,605-parameter model at 99.54% accuracy, directly supporting one more application of the same redundancy; by contrast, extending `ln2` to a fourth fixed coordinate failed at 80.27%.

<<<<<<< SEARCH
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
        )
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=3,
        )
>>>>>>> REPLACE