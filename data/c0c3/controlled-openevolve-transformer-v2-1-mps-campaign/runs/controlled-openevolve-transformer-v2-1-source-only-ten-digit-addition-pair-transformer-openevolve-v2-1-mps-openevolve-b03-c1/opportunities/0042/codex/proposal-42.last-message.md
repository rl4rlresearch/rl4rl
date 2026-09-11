MECHANISM: Per-head query–key gauge-fixed coordinate charts

HYPOTHESIS: Fixing four well-conditioned key coordinates to the identity in each attention head will reduce learned parameters from 1,223 to 1,191 while retaining at least 99% accuracy, because compensating transformations of the corresponding queries and query biases preserve every initialized attention score exactly without reducing query/key rank.

INTENDED_EDIT: For each head, select an invertible 4-by-4 key pivot, absorb it into the query projection and bias, store only the remaining 4-by-3 key coefficients, and reconstruct the full key projection during forward passes.

EVIDENCE: Removing the 36-dimensional rank-six lexical factorization gauge preserved 99.96% accuracy at 1,223 parameters; this applies the same best-conditioned coordinate-chart strategy to the exact per-head query–key change-of-basis redundancy while preserving the full routing capacity whose importance was demonstrated by the failed shared-key design.

<<<<<<< SEARCH
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
=======
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis

            # Query/key scores are invariant under an invertible change of
            # coordinates within each head. Choose a well-conditioned key
            # pivot, fix it to identity, and absorb it into the query and its
            # bias. Only the three non-pivot key columns remain learned.
            q_heads = q_coeff.view(
                self.n_head, self.head_dim, d_model - 1
            )
            k_heads = k_coeff.view(
                self.n_head, self.head_dim, d_model - 1
            )
            q_bias = self.qkv.bias.view(self.n_head, self.head_dim)
            candidates = torch.combinations(
                torch.arange(d_model - 1, device=weight.device),
                r=self.head_dim,
            )
            all_columns = torch.arange(d_model - 1, device=weight.device)
            fixed_q = []
            fixed_q_bias = []
            key_tail = []
            key_inverse_order = []
            for head in range(self.n_head):
                candidate_matrices = k_heads[head][:, candidates].permute(
                    1, 0, 2
                )
                pivot_columns = candidates[
                    torch.linalg.det(candidate_matrices).abs().argmax()
                ]
                tail_columns = all_columns[
                    (
                        all_columns[:, None]
                        != pivot_columns[None, :]
                    ).all(dim=1)
                ]
                column_order = torch.cat((pivot_columns, tail_columns))
                pivot = k_heads[head][:, pivot_columns]
                fixed_q.append(pivot.T @ q_heads[head])
                fixed_q_bias.append(q_bias[head] @ pivot)
                fixed_key = torch.linalg.solve(pivot, k_heads[head])
                key_tail.append(fixed_key[:, tail_columns])
                key_inverse_order.append(torch.argsort(column_order))

            self.qkv.bias = nn.Parameter(
                torch.stack(fixed_q_bias).reshape(-1).clone()
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        torch.stack(fixed_q).reshape(-1),
                        torch.stack(key_tail).reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
            self.register_buffer(
                "key_inverse_order", torch.stack(key_inverse_order)
            )
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
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            d_model, d_model - 1
        )
=======
        matrix_size = d_model * (d_model - 1)
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T

        key_tail_width = d_model - 1 - self.head_dim
        key_tail_size = d_model * key_tail_width
        key_tail = self.qkv.weight[
            matrix_size : matrix_size + key_tail_size
        ].view(self.n_head, self.head_dim, key_tail_width)
        key_pivot = torch.eye(
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        ).expand(self.n_head, -1, -1)
        key_ordered = torch.cat((key_pivot, key_tail), dim=2)
        key_indices = self.key_inverse_order[:, None, :].expand(
            -1, self.head_dim, -1
        )
        k_coeff = torch.gather(key_ordered, 2, key_indices).reshape(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[
            matrix_size + key_tail_size :
        ].view(d_model, d_model - 1)
>>>>>>> REPLACE