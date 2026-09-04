MECHANISM: Residual-subspace rotational gauge fixing

HYPOTHESIS: Combining the qualified three-coordinate shared query bias with one fixed output-projection bias coordinate will retain at least 99% accuracy after 45,000 steps with 1,346 parameters.

INTENDED_EDIT: Adopt the proven three-parameter shared query bias and use the orthogonal freedom in the two token-free residual channels to fix the final attention projection-bias coordinate at zero.

EVIDENCE: The three-parameter shared query bias achieved 100% accuracy with 1,347 parameters, while reducing it to two parameters failed at 74.7%; this instead removes an independent parameter through the residual stream’s two-dimensional rotational symmetry.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit the
        # complete query-bias vector to be shared across the two heads.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - self.head_dim))
        self.proj = nn.Linear(d_model, d_model)
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit the
        # query bias to be shared across heads with one coordinate fixed at
        # zero. Constructing the original Linear first preserves its RNG use.
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Token codes occupy only the first d_model - 2 channels. An orthogonal
        # rotation of the remaining two residual channels can align this bias
        # with one axis, allowing its final coordinate to be fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        query_bias = self.qkv.bias.repeat(self.n_head)
        qkv_bias = torch.cat(
=======
        bsz, seqlen, d_model = x.shape
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = F.pad(self.proj.bias, (0, 1))
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE