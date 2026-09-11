MECHANISM: Gauge-fixed shared-value chart with fused compact projection

HYPOTHESIS: Directly evaluating the fixed 0.02-scaled identity value block and fusing the remaining query, key, and value projections will finish 45,000 updates and retain at least 99% accuracy with 1,126 learned parameters.

INTENDED_EDIT: Replace the shared value dictionary with a fixed full-rank 4×4 block plus 16 learned coefficients, restore the unconstrained attention-output projection, adopt the qualified four-scalar token constraint, and evaluate compact QKV through one linear operation.

EVIDENCE: The 1,138-parameter factorized design achieved 99.78%; the prior 1,126-parameter full-rank value-chart attempt timed out without an accuracy failure, motivating a computationally cheaper direct evaluation of the same capacity-preserving gauge constraint.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates remain
        # gauge-fixed. Constructing the original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_kv_weight_size = 2 * self.head_dim * d_model
        self.qkv.weight = nn.Parameter(
            torch.empty(query_weight_size + shared_kv_weight_size)
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Retain only six learned coordinates of the additive attention-output
        # bias; its final two coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries share one key/value dictionary. The value
        # dictionary uses the full-rank chart [0.02 I | W], removing its
        # sixteen-dimensional basis gauge without removing a value route.
        # Constructing the original Linear preserves constructor RNG use.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * self.head_dim
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_input_start = self.head_dim
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # The full output projection absorbs the fixed value-basis chart.
        # Its final two additive bias coordinates remain fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
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
            (query_weight_size, kv_weight_size, kv_weight_size),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = value_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        key_weight_size = self.head_dim * d_model
        value_weight_size = self.head_dim * self.head_dim
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (query_weight_size, key_weight_size, value_weight_size),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model)
        value_weight = F.pad(
            value_weight.view(self.head_dim, self.head_dim),
            (self.head_dim, 0),
        )
        compact_weight = torch.cat(
            (query_weight, key_weight, value_weight),
            dim=0,
        )
        projected = F.linear(x, compact_weight)
        q, k, v = torch.split(
            projected,
            (d_model, self.head_dim, self.head_dim),
            dim=-1,
        )
        q = q + query_bias
        v = v + 0.02 * x[..., : self.head_dim]
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim].flatten(),
                        )
                    )
=======
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim, module.shared_value_input_start :].flatten(),
                        )
                    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Choose a basis for the tied token subspace by fixing its final three
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 3)
        )
        self.token_emb.fixed_weight_trim = (0, 3)
=======
        # Choose a basis for the tied token subspace by fixing its final four
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 3)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
>>>>>>> REPLACE