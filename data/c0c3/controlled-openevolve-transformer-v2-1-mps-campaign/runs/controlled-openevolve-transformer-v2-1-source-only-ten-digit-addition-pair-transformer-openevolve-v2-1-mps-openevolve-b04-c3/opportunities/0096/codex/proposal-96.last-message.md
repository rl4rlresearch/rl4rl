MECHANISM: Three-scalar output-projection gauge constraint with factorized shared-KV evaluation

HYPOTHESIS: Fixing a third trailing attention-output projection coefficient will reduce the model to 1,138 learned parameters while retaining at least 99% accuracy after 45,000 updates; computing shared keys and values before head replication will reduce redundant work that contributed to the previous verification timeout.

INTENDED_EDIT: Reconstruct three trailing output-projection weights as zeros, preserve full constructor RNG consumption, and evaluate the mathematically equivalent shared key/value projections before duplicating them across heads.

EVIDENCE: The verified 1,139-parameter model achieved 100% accuracy with two fixed output-projection coefficients. The three-coefficient attempt timed out rather than producing an accuracy failure, so repeating that incremental constraint with less redundant attention computation directly tests the unresolved capacity hypothesis.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # Fix two trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 2))
        self.proj.fixed_weight_trim = (0, 2)
=======
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
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model)
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model)
        q = F.linear(x, query_weight, query_bias)
        shared_kv = F.linear(
            x,
            torch.cat((key_weight, value_weight), dim=0),
        )
        k, v = shared_kv.chunk(2, dim=-1)
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = F.pad(self.proj.weight, (0, 2)).view(d_model, d_model)
=======
        proj_weight = F.pad(self.proj.weight, (0, 3)).view(d_model, d_model)
>>>>>>> REPLACE