MECHANISM: Natural-scale per-head query–key gauge charts

HYPOTHESIS: Fixing each head’s key pivot to its original initialized matrix will reduce learned parameters from 1,223 to 1,191 while retaining at least 99% accuracy, because it removes the same exact 32-dimensional query–key gauge as the failed identity chart without rescaling the initialized query and key tensors.

INTENDED_EDIT: Select a well-conditioned four-column key pivot per head, preserve that initialized pivot as a fixed buffer, learn only the remaining key coefficients, and reconstruct the full key projections during forward passes.

EVIDENCE: The identity-pivot query–key chart reached only 95.92% at 1,191 parameters despite preserving initial attention scores, whereas the analogous lexical gauge chart reached 99.96%; retaining the key pivots at their small natural initialization directly tests whether the failure was caused by the identity chart’s severe query/key rescaling rather than the gauge reduction itself.

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

            # Each head admits an invertible query/key change of basis. Freeze
            # a well-conditioned key pivot at its initialized natural scale,
            # leaving the initialized q and k tensors completely unchanged.
            key_heads = k_coeff.view(
                self.n_head, self.head_dim, d_model - 1
            )
            candidates = torch.combinations(
                torch.arange(
                    d_model - 1,
                    device=key_heads.device,
                ),
                r=self.head_dim,
            )
            all_columns = torch.arange(
                d_model - 1, device=key_heads.device
            )
            key_pivots = []
            key_tails = []
            key_inverse_orders = []
            for head in range(self.n_head):
                candidate_matrices = key_heads[head][
                    :, candidates
                ].permute(1, 0, 2)
                pivot_columns = candidates[
                    torch.linalg.det(candidate_matrices).abs().argmax()
                ]
                tail_columns = all_columns[
                    (
                        all_columns[:, None]
                        != pivot_columns[None, :]
                    ).all(dim=1)
                ]
                column_order = torch.cat(
                    (pivot_columns, tail_columns)
                )
                key_pivots.append(
                    key_heads[head][:, pivot_columns]
                )
                key_tails.append(
                    key_heads[head][:, tail_columns]
                )
                key_inverse_orders.append(
                    torch.argsort(column_order)
                )

            self.register_buffer(
                "key_pivot", torch.stack(key_pivots)
            )
            self.register_buffer(
                "key_inverse_order",
                torch.stack(key_inverse_orders),
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        torch.stack(key_tails).reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
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
        key_tail_size = d_model * (
            d_model - 1 - self.head_dim
        )
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        key_tail = self.qkv.weight[
            matrix_size : matrix_size + key_tail_size
        ].view(
            self.n_head,
            self.head_dim,
            d_model - 1 - self.head_dim,
        )
        key_ordered = torch.cat(
            (self.key_pivot, key_tail), dim=2
        )
        key_inverse_order = self.key_inverse_order[
            :, None, :
        ].expand(-1, self.head_dim, -1)
        k_coeff = torch.gather(
            key_ordered, 2, key_inverse_order
        ).reshape(d_model, d_model - 1)
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[
            matrix_size + key_tail_size :
        ].view(d_model, d_model - 1)
>>>>>>> REPLACE