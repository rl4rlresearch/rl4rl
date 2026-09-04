MECHANISM: Per-head query/key change-of-basis gauge fixing

HYPOTHESIS: Fixing an adaptive four-column key basis to identity in each attention head will reduce the verified 638-parameter model to 606 parameters while retaining at least 99% accuracy, because the transformation preserves every initialized attention score and the model’s full learned query/key function class.

INTENDED_EDIT: Select an invertible key-projection pivot per head, retain only the three non-pivot columns, and apply the compensating transpose transformation to each head’s query projection and bias.

EVIDENCE: The analogous shared-value/output-projection gauge fixing removed 16 parameters and achieved 99.94% accuracy, while the current harmonic-token model achieved 99.74%; this tests another exact attention-internal redundancy without narrowing the load-bearing rank-four lexical representation.

<<<<<<< SEARCH
            self.register_buffer(
                "value_inverse_order", torch.argsort(column_order)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
            self.register_buffer(
                "value_inverse_order", torch.argsort(column_order)
            )

    def gauge_fix_query_key(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            matrix_size = d_model * (d_model - 1)
            q_coeff = self.qkv.weight[:matrix_size].view(
                self.n_head, self.head_dim, d_model - 1
            )
            k_coeff = self.qkv.weight[
                matrix_size : 2 * matrix_size
            ].view(self.n_head, self.head_dim, d_model - 1)
            v_tail = self.qkv.weight[2 * matrix_size :]

            candidates = torch.combinations(
                torch.arange(
                    d_model - 1, device=k_coeff.device
                ),
                r=self.head_dim,
            )
            candidate_matrices = k_coeff[:, :, candidates].permute(
                0, 2, 1, 3
            )
            pivot_columns = candidates[
                torch.linalg.det(candidate_matrices).abs().argmax(dim=1)
            ]
            all_columns = torch.arange(
                d_model - 1, device=k_coeff.device
            ).unsqueeze(0).expand(self.n_head, -1)
            tail_columns = all_columns[
                (
                    all_columns.unsqueeze(2)
                    != pivot_columns.unsqueeze(1)
                ).all(dim=2)
            ].view(self.n_head, -1)
            column_order = torch.cat(
                (pivot_columns, tail_columns), dim=1
            )

            pivot = torch.gather(
                k_coeff,
                2,
                pivot_columns.unsqueeze(1).expand(
                    -1, self.head_dim, -1
                ),
            )
            k_tail = torch.linalg.solve(
                pivot,
                torch.gather(
                    k_coeff,
                    2,
                    tail_columns.unsqueeze(1).expand(
                        -1, self.head_dim, -1
                    ),
                ),
            )
            q_coeff = torch.matmul(
                pivot.transpose(1, 2), q_coeff
            )
            q_bias = self.qkv.bias.view(
                self.n_head, self.head_dim
            )
            q_bias = torch.matmul(
                pivot.transpose(1, 2), q_bias.unsqueeze(-1)
            ).squeeze(-1)

            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_tail.reshape(-1),
                        v_tail,
                    )
                ).clone()
            )
            self.qkv.bias = nn.Parameter(q_bias.reshape(-1).clone())
            self.register_buffer(
                "key_pivot",
                torch.eye(
                    self.head_dim,
                    device=k_coeff.device,
                    dtype=k_coeff.dtype,
                ).unsqueeze(0).expand(
                    self.n_head, -1, -1
                ).clone(),
                persistent=False,
            )
            self.register_buffer(
                "key_inverse_order",
                torch.argsort(column_order, dim=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        matrix_size = d_model * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_tail = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1 - self.head_dim
        )
=======
        matrix_size = d_model * (d_model - 1)
        key_tail_size = (
            self.n_head
            * self.head_dim
            * (d_model - 1 - self.head_dim)
        )
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_tail = self.qkv.weight[
            matrix_size : matrix_size + key_tail_size
        ].view(
            self.n_head,
            self.head_dim,
            d_model - 1 - self.head_dim,
        )
        k_reduced = torch.cat(
            (self.key_pivot, k_tail), dim=2
        )
        k_coeff = torch.gather(
            k_reduced,
            2,
            self.key_inverse_order.unsqueeze(1).expand(
                -1, self.head_dim, -1
            ),
        ).reshape(d_model, d_model - 1)
        k_weight = k_coeff @ self.proj_basis.T
        v_tail = self.qkv.weight[
            matrix_size + key_tail_size :
        ].view(
            self.head_dim, d_model - 1 - self.head_dim
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.attn.gauge_fix_value_proj()
            block.mlp.gauge_fix_fc1()
=======
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.attn.gauge_fix_value_proj()
            block.attn.gauge_fix_query_key()
            block.mlp.gauge_fix_fc1()
>>>>>>> REPLACE