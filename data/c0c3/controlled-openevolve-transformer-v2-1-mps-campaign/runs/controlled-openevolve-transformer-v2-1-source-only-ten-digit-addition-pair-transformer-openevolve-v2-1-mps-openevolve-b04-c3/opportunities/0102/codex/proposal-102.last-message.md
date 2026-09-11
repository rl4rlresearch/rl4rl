MECHANISM: Nonzero output-side value-scale anchoring

HYPOTHESIS: Replacing the essential fourth trailing attention-output coefficient with a fixed nonzero 0.02 anchor will retain at least 99% accuracy while reducing the qualified 1,138-parameter model to 1,137 learned parameters.

INTENDED_EDIT: Adopt the qualified factorized shared-key/value backbone, four-scalar token constraint, one-scalar value constraint, and three output zeros; additionally fix the preceding output coefficient at 0.02 while preserving full-matrix initialization RNG consumption.

EVIDENCE: The 1,138-parameter design reached 99.78%, while zeroing its fourth trailing output coefficient collapsed accuracy to 5.92% and cross-head tying reached only 86.7%. A nonzero anchor preserves that essential route while exploiting reciprocal scaling between the shared value coordinate and its output-projection columns.

<<<<<<< SEARCH
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
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are fixed. Constructing the original
        # Linear preserves constructor RNG consumption.
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
        # Preserve the essential fourth trailing output route with a fixed
        # nonzero scale anchor; the final three coefficients remain zero.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 4))
        self.proj.fixed_weight_trim = (0, 4)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_weight = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_full((1,), 0.02),
                self.proj.weight.new_zeros(3),
            )
        ).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            value_weight[: module.shared_kv_dim].flatten(),
=======
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
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