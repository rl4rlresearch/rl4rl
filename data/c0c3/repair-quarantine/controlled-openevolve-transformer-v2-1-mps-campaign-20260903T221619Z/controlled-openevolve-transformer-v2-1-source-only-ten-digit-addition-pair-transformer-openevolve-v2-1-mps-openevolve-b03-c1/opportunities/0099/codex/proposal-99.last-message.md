MECHANISM: Shared-value/output-projection change-of-basis gauge fixing

HYPOTHESIS: Eliminating the 16-dimensional invertible basis redundancy between the shared value map and attention output projection will reduce the model from 866 to 850 learned parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Adaptively select four value-map pivot columns, fix them to an identity matrix, retain only the remaining coefficients, and compensate exactly in each head’s output-projection block.

EVIDENCE: The adaptive pivot-gauge lexical factorization retained 99.98% accuracy at 881 parameters, while the current width-10 model retained 99.22%; this applies the same verified redundancy-removal principle without changing attention scores, initialization function, or model capacity.

<<<<<<< SEARCH
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )

    def gauge_fix_value_proj(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            matrix_size = d_model * (d_model - 1)
            qk_coeff = self.qkv.weight[: 2 * matrix_size]
            v_coeff = self.qkv.weight[2 * matrix_size :].view(
                self.head_dim, d_model - 1
            )

            candidates = torch.combinations(
                torch.arange(d_model - 1, device=v_coeff.device),
                r=self.head_dim,
            )
            candidate_matrices = v_coeff[:, candidates].permute(1, 0, 2)
            pivot_columns = candidates[
                torch.linalg.det(candidate_matrices).abs().argmax()
            ]
            all_columns = torch.arange(d_model - 1, device=v_coeff.device)
            tail_columns = all_columns[
                (all_columns[:, None] != pivot_columns[None, :]).all(dim=1)
            ]
            column_order = torch.cat((pivot_columns, tail_columns))
            pivot = v_coeff[:, pivot_columns]
            v_tail = torch.linalg.solve(
                pivot, v_coeff[:, tail_columns]
            )

            proj_weight = self.proj.weight.view(
                d_model - 1, self.n_head, self.head_dim
            )
            proj_weight = torch.einsum(
                "ohi,ij->ohj", proj_weight, pivot
            )

            self.qkv.weight = nn.Parameter(
                torch.cat((qk_coeff, v_tail.reshape(-1))).clone()
            )
            self.proj.weight = nn.Parameter(
                proj_weight.reshape(d_model - 1, d_model).clone()
            )
            self.register_buffer(
                "value_pivot",
                torch.eye(
                    self.head_dim,
                    device=v_coeff.device,
                    dtype=v_coeff.dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "value_inverse_order", torch.argsort(column_order)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
=======
        v_tail = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1 - self.head_dim
        )
        v_coeff = torch.cat(
            (self.value_pivot, v_tail), dim=1
        )[:, self.value_inverse_order]
        v_weight = v_coeff @ self.proj_basis.T
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
=======
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.attn.gauge_fix_value_proj()
            block.mlp.gauge_fix_fc1()
>>>>>>> REPLACE