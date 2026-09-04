MECHANISM: Second LayerNorm–MLP scale gauge fixing

HYPOTHESIS: Fixing one second-LayerNorm scale coordinate to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because that scale can be absorbed exactly into the corresponding `fc1` input column.

INTENDED_EDIT: Reuse `ScaleFixedLayerNorm` for the second pre-normalization layer, replacing its eight learned scales with seven learned scales and one fixed unit scale.

EVIDENCE: The structurally equivalent first-LayerNorm gauge fixing achieved 99.92% accuracy with 1607 parameters; the second LayerNorm likewise feeds a dense learned projection, so the same redundancy applies without removing attention or bias capacity.

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = ScaleFixedLayerNorm(cfg.d_model)
>>>>>>> REPLACE