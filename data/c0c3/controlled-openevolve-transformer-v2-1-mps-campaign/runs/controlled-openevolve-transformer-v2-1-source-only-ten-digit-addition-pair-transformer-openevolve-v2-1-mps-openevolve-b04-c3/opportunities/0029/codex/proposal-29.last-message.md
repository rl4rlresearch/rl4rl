MECHANISM: Incremental cross-head query-bias sharing

HYPOTHESIS: Sharing a second corresponding query-bias coordinate across the two attention heads will retain at least 99% accuracy after 75,000 steps while reducing the qualified 1,396-parameter design to 1,395 parameters.

INTENDED_EDIT: Represent the eight query-bias coordinates with six learned scalars by sharing two corresponding coordinates across heads; retain the learned sinusoidal gain and endpoint-only validation.

EVIDENCE: The current one-coordinate-sharing design achieved 99.97% accuracy with 1,396 parameters, substantially above the threshold, while clamping a coordinate achieved only 97.45%; this supports another incremental adaptive sharing constraint rather than fixing a coordinate.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Two query coordinates are shared across the
        # first two heads; independent per-head Q/K rotations permit this
        # incremental gauge constraint while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
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
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:2],
                self.qkv.bias[self.head_dim :],
            )
        )
>>>>>>> REPLACE