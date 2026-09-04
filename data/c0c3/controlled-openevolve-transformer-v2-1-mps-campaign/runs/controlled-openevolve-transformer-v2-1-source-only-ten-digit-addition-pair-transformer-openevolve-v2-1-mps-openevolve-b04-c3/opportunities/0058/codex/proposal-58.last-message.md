MECHANISM: Full-vector cross-head query-bias gauge sharing

HYPOTHESIS: Sharing all four corresponding query-bias coordinates across the two heads will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,349 to 1,348 learned parameters.

INTENDED_EDIT: Extend the qualified three-coordinate sharing constraint to the fourth and final query-bias coordinate, leaving the learned positions, rank-six tied token code, attention weights, and training schedule unchanged.

EVIDENCE: Three-coordinate sharing achieved 99.98% accuracy with 1,349 parameters after both one- and two-coordinate sharing achieved 100%; the remaining corresponding coordinate is governed by the same per-head Q/K basis-change symmetry.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit the
        # complete query-bias vector to be shared across the two heads.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - self.head_dim))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:1],
                self.qkv.bias[self.head_dim :],
            )
        )
=======
        query_bias = self.qkv.bias.repeat(self.n_head)
>>>>>>> REPLACE