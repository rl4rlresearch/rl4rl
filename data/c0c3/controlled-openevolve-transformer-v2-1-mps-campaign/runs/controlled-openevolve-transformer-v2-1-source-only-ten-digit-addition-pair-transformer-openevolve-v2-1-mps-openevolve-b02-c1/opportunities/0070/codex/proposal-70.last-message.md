MECHANISM: Rank-three content routing with rank-four value transport

HYPOTHESIS: Decoupling each attention head’s query/key score rank from four to three will reduce the model from 1,038 to 1,022 parameters while retaining at least 99% accuracy, because positional routing remains independently learned and four-dimensional value transport remains intact.

INTENDED_EDIT: Represent each head’s content-dependent attention scores with three learned query/key factors while preserving four-dimensional values, projections, relative-bias tables, gauge-aware virtual optimization, initialization stream, and decoding behavior.

EVIDENCE: Replacing independent relative-bias tables with affine positional pointers failed at 0%, identifying flexible positional routing as load-bearing, while the current independent-table model reaches 99.67%. The earlier rank-three proposal could not be verified, so whether full-rank content scores are necessary remains untested; this patch implements the decoupling consistently through initialization, factor reconstruction, optimization, and score scaling.

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

        with torch.no_grad():
            key_basis = key_heads[..., : self.head_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.head_dim :]
                )
            )
            self.query_bias.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., self.head_dim :]
                )
            )

        # Consumed by QuotientAdamW to preserve the virtual factor updates.
        self._initial_query_weight = query_weight.detach().clone()
        self._initial_key_weight = key_weight.detach().clone()
        self._initial_query_bias = query_bias.detach().clone()
        self._initial_value_weight = value_weight.detach().clone()
=======
        full_query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        full_key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        query_heads = full_query_heads[:, : self.score_dim]
        key_heads = full_key_heads[:, : self.score_dim]
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        full_bias_heads = query_bias.view(
            self.n_head, self.head_dim
        )
        bias_heads = full_bias_heads[:, : self.score_dim]

        with torch.no_grad():
            key_basis = key_heads[..., : self.score_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.score_dim :]
                )
            )
            self.query_bias.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., self.head_dim :]
                )
            )

        # Consumed by QuotientAdamW to preserve virtual rank-three updates.
        self._initial_query_weight = (
            query_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_key_weight = (
            key_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_query_bias = (
            bias_heads.reshape(-1).detach().clone()
        )
        self._initial_value_weight = value_weight.detach().clone()
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