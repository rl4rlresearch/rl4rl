MECHANISM: Seventh pre-MLP LayerNorm-bias absorption

HYPOTHESIS: Fixing a seventh `ln2` bias coordinate will reduce the verified 1,578-parameter model to 1,577 parameters while retaining at least 99% accuracy, because its effect can be represented by the downstream learned `fc1` bias.

INTENDED_EDIT: Increase `ln2`’s fixed bias coordinates from six to seven, leaving the architecture, training schedule, and protected decoding interface unchanged.

EVIDENCE: The current six-coordinate `ln2` compaction achieved 99.73% accuracy at 1,578 parameters; this tests the next coordinate in that existing gauge family after additional positional-row gauges failed.

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE