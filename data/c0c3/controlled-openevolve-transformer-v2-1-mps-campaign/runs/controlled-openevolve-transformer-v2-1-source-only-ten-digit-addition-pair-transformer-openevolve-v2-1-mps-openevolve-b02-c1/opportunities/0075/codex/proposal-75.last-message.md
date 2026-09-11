MECHANISM: Asymmetric rank-one/rank-two content routing

HYPOTHESIS: Reducing only head 1’s query/key score rank from two to one will lower the model from 1,002 to 990 parameters while retaining at least 99% accuracy, because rank one failed only when imposed on both heads and head 1’s positional pathway tolerated substantially more compression than head 0’s.

INTENDED_EDIT: Preserve rank-two content routing in head 0 while using rank-one routing in head 1, with head-specific score scaling, initialization, and gauge-aware virtual optimization.

EVIDENCE: Two rank-two heads achieved 99.93%, whereas reducing both heads to rank one collapsed to 44.36%; additionally, head 1 tolerated learned boundary sharing and interpolation while analogous head-0 boundary compression failed, motivating an asymmetric ablation that preserves head 0’s capacity.

<<<<<<< SEARCH
        self.score_dim = self.head_dim - 2
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
=======
        self.score_dims = tuple(
            self.head_dim - 2 - int(head == 1)
            for head in range(n_head)
        )
        reduced_width = d_model - 1
        self.query_weight = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(1, score_dim, reduced_width)
                )
                for score_dim in self.score_dims
            ]
        )
        self.key_tail = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        1,
                        score_dim,
                        reduced_width - score_dim,
                    )
                )
                for score_dim in self.score_dims
            ]
        )
        self.query_bias = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, score_dim))
                for score_dim in self.score_dims
            ]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        query_heads = [
            full_query_heads[
                head : head + 1, :score_dim
            ]
            for head, score_dim in enumerate(self.score_dims)
        ]
        key_heads = [
            full_key_heads[
                head : head + 1, :score_dim
            ]
            for head, score_dim in enumerate(self.score_dims)
        ]
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        full_bias_heads = query_bias.view(
            self.n_head, self.head_dim
        )
        bias_heads = [
            full_bias_heads[
                head : head + 1, :score_dim
            ]
            for head, score_dim in enumerate(self.score_dims)
        ]

        with torch.no_grad():
            for (
                query_param,
                key_tail_param,
                bias_param,
                query_head,
                key_head,
                bias_head,
                score_dim,
            ) in zip(
                self.query_weight,
                self.key_tail,
                self.query_bias,
                query_heads,
                key_heads,
                bias_heads,
                self.score_dims,
            ):
                key_basis = key_head[..., :score_dim]
                query_param.copy_(
                    torch.matmul(
                        key_basis.transpose(-1, -2),
                        query_head,
                    )
                )
                key_tail_param.copy_(
                    torch.linalg.solve(
                        key_basis, key_head[..., score_dim:]
                    )
                )
                bias_param.copy_(
                    torch.matmul(
                        key_basis.transpose(-1, -2),
                        bias_head.unsqueeze(-1),
                    ).squeeze(-1)
                )
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., self.head_dim :]
                )
            )

        # Consumed by QuotientAdamW to preserve virtual score updates.
        self._initial_query_weight = [
            head.reshape(-1, self.d_model - 1).detach().clone()
            for head in query_heads
        ]
        self._initial_key_weight = [
            head.reshape(-1, self.d_model - 1).detach().clone()
            for head in key_heads
        ]
        self._initial_query_bias = [
            head.reshape(-1).detach().clone()
            for head in bias_heads
        ]
        self._initial_value_weight = value_weight.detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_prefix = torch.eye(
            self.score_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
        key_weight = torch.cat([key_prefix, self.key_tail], dim=-1)
        q = torch.einsum(
            "btf,hdf->bhtd", reduced_x, self.query_weight
        )
        q = q + self.query_bias.unsqueeze(0).unsqueeze(2)
        k = torch.einsum(
            "bsf,hdf->bhsd", reduced_x, key_weight
        )
        att = torch.einsum("bhtd,bhsd->bhts", q, k)
=======
        score_heads = []
        for (
            score_dim,
            query_weight,
            key_tail,
            query_bias,
        ) in zip(
            self.score_dims,
            self.query_weight,
            self.key_tail,
            self.query_bias,
        ):
            key_prefix = torch.eye(
                score_dim,
                device=reduced_x.device,
                dtype=reduced_x.dtype,
            ).unsqueeze(0)
            key_weight = torch.cat(
                [key_prefix, key_tail], dim=-1
            )
            q = torch.einsum(
                "btf,hdf->bhtd", reduced_x, query_weight
            )
            q = q + query_bias.unsqueeze(0).unsqueeze(2)
            k = torch.einsum(
                "bsf,hdf->bhsd", reduced_x, key_weight
            )
            score_heads.append(
                torch.einsum("bhtd,bhsd->bhts", q, k).squeeze(1)
                / math.sqrt(score_dim)
            )
        att = torch.stack(score_heads, dim=1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att / math.sqrt(self.qkv.score_dim)
        positions = torch.arange(seqlen, device=x.device)
=======
        positions = torch.arange(seqlen, device=x.device)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.score_specs = [
            (
                block.attn.qkv.query_weight,
                block.attn.qkv.key_tail,
                block.attn.qkv.query_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
=======
        self.score_specs = [
            (
                query_weight,
                key_tail,
                query_bias,
                block.attn.qkv,
                head,
            )
            for block in model.blocks
            for head, (
                query_weight,
                key_tail,
                query_bias,
            ) in enumerate(
                zip(
                    block.attn.qkv.query_weight,
                    block.attn.qkv.key_tail,
                    block.attn.qkv.query_bias,
                )
            )
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.score_params = [
            param
            for query_weight, key_tail, query_bias, _
            in self.score_specs
            for param in (query_weight, key_tail, query_bias)
        ]
=======
        self.score_params = [
            param
            for query_weight, key_tail, query_bias, _, _
            in self.score_specs
            for param in (query_weight, key_tail, query_bias)
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.score_states = []
        for query_param, _, _, qkv in self.score_specs:
            query_weight = qkv._initial_query_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            key_weight = qkv._initial_key_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            query_bias = qkv._initial_query_bias.to(
                device=query_param.device, dtype=query_param.dtype
            )
            delattr(qkv, "_initial_query_weight")
            delattr(qkv, "_initial_key_weight")
            delattr(qkv, "_initial_query_bias")

            full_shape = list(query_weight.shape)
            full_shape[1] += 1
            self.score_states.append(
                {
                    "step": 0,
                    "query_weight": query_weight,
                    "key_weight": key_weight,
                    "query_bias": query_bias,
                    "exp_avg_query": query_weight.new_zeros(full_shape),
                    "exp_avg_sq_query": query_weight.new_zeros(full_shape),
                    "exp_avg_key": key_weight.new_zeros(full_shape),
                    "exp_avg_sq_key": key_weight.new_zeros(full_shape),
                    "exp_avg_bias": query_bias.new_zeros(query_bias.shape),
                    "exp_avg_sq_bias": query_bias.new_zeros(
                        query_bias.shape
                    ),
                }
            )

=======
        self.score_states = []
        for query_param, _, _, qkv, head in self.score_specs:
            query_weight = qkv._initial_query_weight[head].to(
                device=query_param.device, dtype=query_param.dtype
            )
            key_weight = qkv._initial_key_weight[head].to(
                device=query_param.device, dtype=query_param.dtype
            )
            query_bias = qkv._initial_query_bias[head].to(
                device=query_param.device, dtype=query_param.dtype
            )

            full_shape = list(query_weight.shape)
            full_shape[1] += 1
            self.score_states.append(
                {
                    "step": 0,
                    "query_weight": query_weight,
                    "key_weight": key_weight,
                    "query_bias": query_bias,
                    "exp_avg_query": query_weight.new_zeros(full_shape),
                    "exp_avg_sq_query": query_weight.new_zeros(full_shape),
                    "exp_avg_key": key_weight.new_zeros(full_shape),
                    "exp_avg_sq_key": key_weight.new_zeros(full_shape),
                    "exp_avg_bias": query_bias.new_zeros(query_bias.shape),
                    "exp_avg_sq_bias": query_bias.new_zeros(
                        query_bias.shape
                    ),
                }
            )
        for block in model.blocks:
            delattr(block.attn.qkv, "_initial_query_weight")
            delattr(block.attn.qkv, "_initial_key_weight")
            delattr(block.attn.qkv, "_initial_query_bias")

>>>>>>> REPLACE

<<<<<<< SEARCH
        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters.
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
=======
        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters.
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                query_param.grad is None
=======
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                query_param.grad is None
>>>>>>> REPLACE