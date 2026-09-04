MECHANISM: Nonzero shared-key gauge anchoring with fused compact QKV evaluation

HYPOTHESIS: Fixing one trailing shared-key coefficient at 0.02 will preserve at least 99% accuracy at 1,137 learned parameters, while fusing the factorized query and shared-key/value projections will allow all 45,000 updates to finish within the verification limit.

INTENDED_EDIT: Adopt the qualified three-scalar attention-output constraint, remove one shared-key scalar and reconstruct it as 0.02, then evaluate the compact query and shared key/value dictionaries with one linear operation.

EVIDENCE: The three-output-constraint design reached 99.78% accuracy at 1,138 parameters. The prior nonzero shared-key anchor timed out rather than failing accuracy, so preserving that unresolved one-parameter gauge probe while reducing projection overhead is the most direct next test.

<<<<<<< SEARCH
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are gauge-fixed. Constructing the
        # original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_weight_trim = 1
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Fix one coefficient on the output side of the shared-value basis.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 1))
        self.proj.fixed_weight_trim = (0, 1)
=======
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are zero-fixed; one trailing key
        # coefficient anchors the reciprocal query/key scale at 0.02.
        # Constructing the original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model - 1
        shared_value_weight_size = self.head_dim * d_model - 1
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_key_weight_trim = 1
        self.qkv.shared_value_weight_trim = 1
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Fix three trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 3))
        self.proj.fixed_weight_trim = (0, 3)
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size - self.qkv.shared_key_weight_trim,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = F.pad(
            key_weight,
            (0, self.qkv.shared_key_weight_trim),
            value=0.02,
        ).view(self.head_dim, d_model)
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model)
        compact_weight = torch.cat(
            (query_weight, key_weight, value_weight),
            dim=0,
        )
        compact_bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * self.head_dim))
        )
        q, k, v = F.linear(x, compact_weight, compact_bias).split(
            (d_model, self.head_dim, self.head_dim),
            dim=-1,
        )
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = F.pad(self.proj.weight, (0, 1)).view(d_model, d_model)
=======
        proj_weight = F.pad(self.proj.weight, (0, 3)).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
=======
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_key_weight_trim],
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
>>>>>>> REPLACE