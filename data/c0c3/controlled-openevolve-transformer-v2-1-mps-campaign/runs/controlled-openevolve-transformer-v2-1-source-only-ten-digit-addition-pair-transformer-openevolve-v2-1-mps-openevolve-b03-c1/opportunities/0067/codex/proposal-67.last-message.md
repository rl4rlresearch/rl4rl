MECHANISM: Single-coordinate query/key latent-gauge fixing

HYPOTHESIS: Fixing one key-projection coefficient through a compensating query/key basis transformation will reduce the model from 1,119 to 1,118 learned parameters while retaining at least 99% accuracy, because it preserves the initialized attention scores and removes only one exact latent-factorization degree of freedom.

INTENDED_EDIT: Apply an invertible row transformation to the first head’s key projection that zeros one coefficient, apply the inverse-transpose transformation to its query projection and bias, omit the fixed zero from the learned parameter vector, and reconstruct it during forward passes.

EVIDENCE: The 1,119-parameter design achieved 99.53%, while further second-head relative-bias pruning fell to 73.42% and MLP-bias gauge compression collapsed; this motivates preserving all demonstrated routing and MLP parameters while cautiously testing a single exact query/key factorization gauge.

<<<<<<< SEARCH
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            # Combine the initialized per-head value maps at variance-preserving
            # scale, then learn a single semantic readout used by both routes.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
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
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = (q_weight @ self.proj_basis).view(
                self.n_head, self.head_dim, d_model - 1
            )
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = (k_weight @ self.proj_basis).view(
                self.n_head, self.head_dim, d_model - 1
            )

            # Query/key dot products are invariant under an invertible change
            # of their shared latent basis. Use a well-scaled shear to zero one
            # key coefficient and apply its inverse transpose to the matching
            # query map, preserving the initialized attention scores.
            gauge_head = 0
            head_keys = k_coeff[gauge_head].clone()
            pivot_flat = head_keys.abs().argmax()
            pivot_row = pivot_flat // (d_model - 1)
            pivot_column = pivot_flat % (d_model - 1)
            target_row = (pivot_row + 1) % self.head_dim
            transform = torch.eye(
                self.head_dim,
                device=head_keys.device,
                dtype=head_keys.dtype,
            )
            transform[target_row, pivot_row] = (
                -head_keys[target_row, pivot_column]
                / head_keys[pivot_row, pivot_column]
            )
            k_coeff[gauge_head] = transform @ head_keys
            q_coeff[gauge_head] = torch.linalg.solve(
                transform.T, q_coeff[gauge_head]
            )
            q_bias = self.qkv.bias.view(
                self.n_head, self.head_dim
            ).clone()
            q_bias[gauge_head] = torch.linalg.solve(
                transform.T, q_bias[gauge_head]
            )
            self.qkv.bias = nn.Parameter(q_bias.reshape(-1).clone())

            fixed_index = (
                gauge_head * self.head_dim * (d_model - 1)
                + target_row * (d_model - 1)
                + pivot_column
            )
            self.register_buffer(
                "k_gauge_index", fixed_index.to(dtype=torch.long)
            )
            k_coeff = k_coeff.reshape(-1)
            k_mask = (
                torch.arange(k_coeff.numel(), device=k_coeff.device)
                != self.k_gauge_index
            )

            # Combine the initialized per-head value maps at variance-preserving
            # scale, then learn a single semantic readout used by both routes.
            v_weight = v_weight.view(
                self.n_head, self.head_dim, d_model
            ).sum(dim=0) / math.sqrt(self.n_head)
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff[k_mask],
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1
        )
=======
        k_learned = self.qkv.weight[
            matrix_size : 2 * matrix_size - 1
        ]
        k_mask = (
            torch.arange(matrix_size, device=x.device)
            != self.k_gauge_index
        )
        k_coeff = k_learned.new_zeros(matrix_size).masked_scatter(
            k_mask, k_learned
        ).view(d_model, d_model - 1)
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[2 * matrix_size - 1 :].view(
            self.head_dim, d_model - 1
        )
>>>>>>> REPLACE