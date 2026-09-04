MECHANISM: Complementary second-head query–key gauge fixing

HYPOTHESIS: Fixing one initialized query coefficient in the second attention head will reduce the model from 1,044 to 1,043 parameters while retaining at least 99% accuracy, because reciprocal query–key scaling is an exact independent symmetry for each head.

INTENDED_EDIT: Store the largest initialized second-head query coefficient as a buffer, learn all remaining query coefficients, and reconstruct the complete query map during attention while preserving the existing first-head key gauge fix.

EVIDENCE: Fixing the first head’s analogous key-scale gauge achieved 99.99% accuracy at 1,044 parameters. The prior second-head key attempt failed the learned-model requirement rather than accuracy, so applying the independent symmetry through the complementary query map tests whether that failure was implementation-specific.

<<<<<<< SEARCH
            # Reciprocally scaling one head's query and key maps leaves its
            # attention scores unchanged. Fix the largest initialized key
            # coefficient in the first head and learn every other coordinate.
            k_flat = k_coeff.reshape(-1)
            first_head_size = self.head_dim * (d_model - 1)
            k_pivot_index = k_flat[:first_head_size].abs().argmax()
            k_all_coordinates = torch.arange(
                k_flat.numel(), device=k_flat.device
            )
            k_tail_coordinates = k_all_coordinates[
                k_all_coordinates != k_pivot_index
            ]
            k_coordinate_order = torch.cat(
                (k_pivot_index.unsqueeze(0), k_tail_coordinates)
            )
            self.register_buffer(
                "k_pivot", k_flat[k_pivot_index].detach().clone()
            )
            self.register_buffer(
                "k_inverse_order", torch.argsort(k_coordinate_order)
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_flat[k_tail_coordinates],
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )
=======
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and fix
            # one query coordinate in the complementary second head.
            q_flat = q_coeff.reshape(-1)
            second_head_start = self.head_dim * (d_model - 1)
            q_pivot_index = second_head_start + q_flat[
                second_head_start:
            ].abs().argmax()
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                q_all_coordinates != q_pivot_index
            ]
            q_coordinate_order = torch.cat(
                (q_pivot_index.unsqueeze(0), q_tail_coordinates)
            )
            self.register_buffer(
                "q_pivot", q_flat[q_pivot_index].detach().clone()
            )
            self.register_buffer(
                "q_inverse_order", torch.argsort(q_coordinate_order)
            )

            k_flat = k_coeff.reshape(-1)
            first_head_size = self.head_dim * (d_model - 1)
            k_pivot_index = k_flat[:first_head_size].abs().argmax()
            k_all_coordinates = torch.arange(
                k_flat.numel(), device=k_flat.device
            )
            k_tail_coordinates = k_all_coordinates[
                k_all_coordinates != k_pivot_index
            ]
            k_coordinate_order = torch.cat(
                (k_pivot_index.unsqueeze(0), k_tail_coordinates)
            )
            self.register_buffer(
                "k_pivot", k_flat[k_pivot_index].detach().clone()
            )
            self.register_buffer(
                "k_inverse_order", torch.argsort(k_coordinate_order)
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_flat[q_tail_coordinates],
                        k_flat[k_tail_coordinates],
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_coeff = self.qkv.weight[:matrix_size].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[matrix_size : 2 * matrix_size - 1],
            )
        )
        k_coeff = k_flat[self.k_inverse_order].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight[2 * matrix_size - 1 :],
            )
        )
=======
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight[: matrix_size - 1],
            )
        )
        q_coeff = q_flat[self.q_inverse_order].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[
                    matrix_size - 1 : 2 * matrix_size - 2
                ],
            )
        )
        k_coeff = k_flat[self.k_inverse_order].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight[2 * matrix_size - 2 :],
            )
        )
>>>>>>> REPLACE