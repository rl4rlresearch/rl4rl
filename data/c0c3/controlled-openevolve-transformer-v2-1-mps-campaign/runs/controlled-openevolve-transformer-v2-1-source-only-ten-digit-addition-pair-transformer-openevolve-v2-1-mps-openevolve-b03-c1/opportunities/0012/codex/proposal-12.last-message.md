MECHANISM: Absorb query offset into the learned pre-attention bias

HYPOTHESIS: Removing the explicit eight-parameter query bias will reduce the model from 1,604 to 1,596 parameters while retaining at least 99% accuracy, because the learned `ln1_bias` can produce query offsets through the query projection, the retained value bias can compensate its value-side effect, and its key-side effect is softmax-invariant.

INTENDED_EDIT: Retain only the value portion of the RNG-aligned QKV bias and stop adding a separate query bias.

EVIDENCE: The 1,604-parameter design achieved 99.37% accuracy; removing value bias collapsed accuracy to 30.85%, while removing `ln1` bias reached only 75.82%, motivating preservation of both and elimination of the query bias they can jointly represent.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query and value portions of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
=======
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the value portion of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias
=======
        q, k, v = qkv.chunk(3, dim=-1)
        v = v + self.qkv.bias
>>>>>>> REPLACE