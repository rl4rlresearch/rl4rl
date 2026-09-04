MECHANISM: Sixth downstream-affine LayerNorm-bias gauge

HYPOTHESIS: Reproducing the verified 1,590-parameter design by fixing a sixth `ln2` bias coordinate will retain at least 99% accuracy because `fc1`’s independent biases absorb the removed LayerNorm offset.

INTENDED_EDIT: Increase `ln2`’s fixed bias coordinates from five to six, reducing the learned parameter count from 1,591 to 1,590.

EVIDENCE: The identical six-coordinate design previously achieved 99.88% accuracy with 1,590 parameters, while fixing a seventh coordinate fell to 97.17%.

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=5,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
        )
>>>>>>> REPLACE