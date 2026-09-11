MECHANISM: Three-scalar output-side shared-value basis constraint

HYPOTHESIS: Extending the verified 1,139-parameter design by fixing a third trailing attention-output projection coefficient at zero will preserve at least 99% accuracy after 45,000 updates while reducing the model to 1,138 learned parameters.

INTENDED_EDIT: Adopt the qualified four-scalar token-embedding and one-scalar shared-value constraints, then reconstruct the final three attention-output projection weights as zeros while preserving full constructor RNG consumption.

EVIDENCE: Reference Design 2 achieved 100% accuracy with one fixed shared-value coefficient and two fixed attention-output projection coefficients at 1,139 parameters; extending the independently successful output-side constraint by one scalar is the smallest informative next probe.

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
        # Fix three trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 3))
        self.proj.fixed_weight_trim = (0, 3)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (query_weight_size, kv_weight_size, kv_weight_size),
        )
=======
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_weight = value_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
=======
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_weight = F.pad(self.proj.weight, (0, 3)).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
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
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim].flatten(),
=======
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 3)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
>>>>>>> REPLACE