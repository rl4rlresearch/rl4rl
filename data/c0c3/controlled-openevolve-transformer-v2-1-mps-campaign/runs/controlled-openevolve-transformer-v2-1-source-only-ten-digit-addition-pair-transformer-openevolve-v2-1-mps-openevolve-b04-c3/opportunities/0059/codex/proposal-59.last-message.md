MECHANISM: One-coordinate shared-query basis gauge

HYPOTHESIS: Fixing one coordinate of the fully shared query-bias vector at zero will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,348 to 1,347 parameters.

INTENDED_EDIT: Adopt full cross-head query-bias sharing and learn only three of its four coordinates, padding the fourth with zero while preserving the qualified learned-position and rank-six token backbone.

EVIDENCE: Reference Design 3 achieved 100% accuracy with 1,348 parameters and a fully shared four-coordinate query bias; the Q/K basis symmetry motivates testing a single zero-coordinate gauge while retaining three adaptable bias coordinates and the successful initialization.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes allow the
        # query bias to be shared across heads and one shared coordinate to be
        # fixed at zero. Construct the original Linear first to preserve its
        # constructor RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE