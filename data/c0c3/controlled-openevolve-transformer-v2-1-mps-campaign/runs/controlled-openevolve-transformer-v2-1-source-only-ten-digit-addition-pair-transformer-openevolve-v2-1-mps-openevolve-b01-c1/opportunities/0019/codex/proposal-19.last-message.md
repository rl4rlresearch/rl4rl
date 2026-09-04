MECHANISM: Incremental attention LayerNorm bias absorption

HYPOTHESIS: Fixing a second `ln1` bias coordinate to zero will reduce parameters from 1,594 to 1,593 while retaining at least 99% accuracy, because its effect can be absorbed by the learned QKV and attention-output biases.

INTENDED_EDIT: Use the existing two-coordinate-pruned LayerNorm before causal self-attention, leaving all other model capacity, initialization, and training settings unchanged.

EVIDENCE: Pruning one `ln1` bias coordinate achieved 99.62% accuracy at 1,594 parameters, while two-coordinate `ln2` bias pruning achieved 99.97%; this is the smallest incremental test of the same successful bias-absorption mechanism.

<<<<<<< SEARCH
        self.ln1 = OnePrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = TwoPrunedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE