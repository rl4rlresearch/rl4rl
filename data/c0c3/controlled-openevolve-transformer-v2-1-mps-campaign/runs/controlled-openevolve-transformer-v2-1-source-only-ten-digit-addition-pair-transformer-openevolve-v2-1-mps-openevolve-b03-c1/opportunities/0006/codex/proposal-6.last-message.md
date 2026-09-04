MECHANISM: Remove output-bias-redundant value bias

HYPOTHESIS: Removing the eight value-projection bias parameters will reduce the model from 1,628 to 1,620 parameters while retaining at least 99% accuracy, because attention weights sum to one and therefore the value bias contributes a position-independent vector that the following output-projection bias can represent exactly.

INTENDED_EDIT: Retain only the learned query bias in the RNG-aligned combined QKV module and stop adding a separate value bias.

EVIDENCE: The current RNG-aligned 1,628-parameter design achieved 99.77% accuracy, and the successful LayerNorm-bias removal shows that eliminating exactly absorbable affine parameters can preserve accuracy without narrowing the capacity-critical feed-forward network.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query and value portions of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
=======
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query bias. A value bias passes unchanged through
        # attention because its weights sum to one, so the projection bias absorbs it.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias
=======
        q = q + self.qkv.bias
>>>>>>> REPLACE