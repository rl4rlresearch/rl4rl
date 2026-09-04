MECHANISM: Distributed LayerNorm bias anchoring

HYPOTHESIS: Anchoring one bias coordinate in the pre-MLP LayerNorm will reduce the model from 1,643 to 1,642 parameters while retaining at least 99% accuracy, because the following learned `fc1` bias can absorb the omitted constant offset without further constraining the already-sensitive pre-attention normalization.

INTENDED_EDIT: Reuse `AnchoredLayerNorm` for `ln2`, preserving all attention and feed-forward widths.

EVIDENCE: One-coordinate pre-attention anchoring achieved 99.95% with 1,643 parameters, while anchoring a second coordinate in that same normalization collapsed accuracy; distributing the next one-coordinate reduction to the independently absorbable pre-MLP bias is the smallest informative alternative.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = AnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE