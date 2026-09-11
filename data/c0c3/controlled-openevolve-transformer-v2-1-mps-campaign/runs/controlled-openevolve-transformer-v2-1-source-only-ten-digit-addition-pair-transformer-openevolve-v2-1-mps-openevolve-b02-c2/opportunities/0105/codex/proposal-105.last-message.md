MECHANISM: Seventh downstream-affine LayerNorm-bias gauge

HYPOTHESIS: Fixing a seventh `ln2` bias coordinate will reduce the verified 1,590-parameter model to 1,589 parameters while retaining at least 99% accuracy, because `fc1`’s independent biases can absorb the removed LayerNorm offset.

INTENDED_EDIT: Increase `ln2`’s fixed bias coordinates from six to seven while preserving the qualified architecture and training configuration.

EVIDENCE: Successive fourth, fifth, and sixth `ln2` bias reductions achieved 99.90%, 99.89%, and 99.88% accuracy respectively; extending this repeatedly successful local redundancy is better supported than the failed third-`fc2` column, second-`fc1` row, attention-projection column, or Q/K rotation reductions.

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
        )
>>>>>>> REPLACE