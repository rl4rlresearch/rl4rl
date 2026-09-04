MECHANISM: Single-scalar shared-value basis gauge fixing

HYPOTHESIS: The verified 1,142-parameter model will retain at least 99% accuracy after 45,000 updates when one trailing shared-value projection coefficient is fixed at zero, yielding 1,141 parameters.

INTENDED_EDIT: Adopt the qualified four-scalar token constraint and remove one learned shared-value coefficient, reconstructing it as zero while preserving full initialization RNG consumption.

EVIDENCE: The four-scalar token design achieved 100% accuracy at 1,142 parameters. Fixing an entire 16-coefficient value block failed, motivating a minimal one-scalar probe that retains the shared value projection’s rank and flexible learned output projection.

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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim].flatten(),
=======
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Choose a basis for the tied token subspace by fixing its final two scalars.
        # The full Embedding constructor above preserves constructor RNG use.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 2)
        )
        self.token_emb.fixed_weight_trim = (0, 2)
=======
        # Choose a basis for the tied token subspace by fixing its final four
        # scalars. The full Embedding above preserves constructor RNG use.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 2)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
>>>>>>> REPLACE