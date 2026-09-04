MECHANISM: Decoupled three-dimensional content routing with four-dimensional value heads

HYPOTHESIS: Reducing each head’s learned query/key score rank from four to three will lower the model from 1,042 to 1,026 parameters while retaining at least 99% accuracy, because the evidence identifies flexible independent relative-position tables—not full-rank content scores—as load-bearing.

INTENDED_EDIT: Decouple attention-score width from value-head width, retaining four-dimensional learned values but using three-dimensional learned query/key factors and the corresponding scale.

EVIDENCE: Replacing the independent relative-bias tables with affine positional pointers failed at 0%, while the current independent-table model achieved 99.38%; preserving those tables and compressing only the orthogonal content-routing pathway cleanly tests the shared assumption that standard attention must couple score rank to value width.

<<<<<<< SEARCH
        self.head_dim = d_model // n_head
        reduced_width = d_model - 1
        self.query_weight = nn.Parameter(
            torch.empty(n_head, self.head_dim, reduced_width)
        )
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.head_dim)
        )
=======
        self.head_dim = d_model // n_head
        self.score_dim = self.head_dim - 1
        reduced_width = d_model - 1
        self.query_weight = nn.Parameter(
            torch.empty(n_head, self.score_dim, reduced_width)
        )
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.score_dim,
                reduced_width - self.score_dim,
            )
        )
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.score_dim)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_query_weight = full_weight[: self.d_model]
        full_key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        effective_value_weight = (
            value_weight[:, :-1] - value_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(self.d_model)
        query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        bias_heads = query_bias.view(self.n_head, self.head_dim)
=======
        full_query_weight = full_weight[: self.d_model].view(
            self.n_head, self.head_dim, self.d_model
        )[:, : self.score_dim].reshape(-1, self.d_model)
        full_key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ].view(
            self.n_head, self.head_dim, self.d_model
        )[:, : self.score_dim].reshape(-1, self.d_model)
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        effective_value_weight = (
            value_weight[:, :-1] - value_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(
            self.n_head * self.score_dim
        )
        query_heads = query_weight.view(
            self.n_head, self.score_dim, self.d_model - 1
        )
        key_heads = key_weight.view(
            self.n_head, self.score_dim, self.d_model - 1
        )
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        bias_heads = query_bias.view(self.n_head, self.score_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_prefix = torch.eye(
            self.head_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
=======
        key_prefix = torch.eye(
            self.score_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att / math.sqrt(self.head_dim)
=======
        att = att / math.sqrt(self.qkv.score_dim)
>>>>>>> REPLACE