MECHANISM: Residual second-head orthogonal query–key gauge fixing

HYPOTHESIS: Eliminating another second-head query coefficient through a score-preserving orthogonal query–key rotation will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the second and third features of the second attention head, omit the resulting fixed-zero query coefficient from the packed QKV parameter, and reconstruct it during attention.

EVIDENCE: The first two orthogonal gauge removals preserved attention scores and achieved 99.93% and 99.90% accuracy. The attempted additional first-head removal timed out without adverse accuracy evidence, so the symmetric second-head test is the most direct implementation-specific follow-up.

<<<<<<< SEARCH
            q_coeff[
                self.head_dim, second_rotation_column
            ] = 0.0
            # Combine the initialized per-head value maps at variance-preserving
=======
            q_coeff[
                self.head_dim, second_rotation_column
            ] = 0.0

            # The first rotated row is now fixed, leaving an independent
            # rotation of the next two second-head features. Use it to remove
            # one more query coordinate without changing any attention score.
            third_rotation_column = torch.linalg.vector_norm(
                q_coeff[
                    self.head_dim + 1 : self.head_dim + 3
                ],
                dim=0,
            ).argmax()
            a = q_coeff[
                self.head_dim + 1, third_rotation_column
            ]
            b = q_coeff[
                self.head_dim + 2, third_rotation_column
            ]
            radius = torch.sqrt(a.square() + b.square())
            third_rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[
                self.head_dim + 1 : self.head_dim + 3
            ] = (
                third_rotation
                @ q_coeff[
                    self.head_dim + 1 : self.head_dim + 3
                ]
            )
            k_coeff[
                self.head_dim + 1 : self.head_dim + 3
            ] = (
                third_rotation
                @ k_coeff[
                    self.head_dim + 1 : self.head_dim + 3
                ]
            )
            self.qkv.bias[
                self.head_dim + 1 : self.head_dim + 3
            ].copy_(
                third_rotation
                @ self.qkv.bias[
                    self.head_dim + 1 : self.head_dim + 3
                ]
            )
            q_coeff[
                self.head_dim + 1, third_rotation_column
            ] = 0.0
            # Combine the initialized per-head value maps at variance-preserving
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and the
            # complementary second-head query pivot, while excluding both
            # orthogonally fixed zero query coordinates.
            q_flat = q_coeff.reshape(-1)
            second_head_start = self.head_dim * (d_model - 1)
            q_pivot_index = second_head_start + q_flat[
                second_head_start:
            ].abs().argmax()
            first_q_zero_index = rotation_column
            second_q_zero_index = (
                second_head_start + second_rotation_column
            )
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                (q_all_coordinates != q_pivot_index)
                & (q_all_coordinates != first_q_zero_index)
                & (q_all_coordinates != second_q_zero_index)
            ]
            q_coordinate_order = torch.cat(
                (
                    q_pivot_index.unsqueeze(0),
                    first_q_zero_index.unsqueeze(0),
                    second_q_zero_index.unsqueeze(0),
                    q_tail_coordinates,
                )
            )
=======
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and the
            # complementary second-head query pivot, while excluding all three
            # orthogonally fixed zero query coordinates.
            q_flat = q_coeff.reshape(-1)
            second_head_start = self.head_dim * (d_model - 1)
            q_pivot_index = second_head_start + q_flat[
                second_head_start:
            ].abs().argmax()
            first_q_zero_index = rotation_column
            second_q_zero_index = (
                second_head_start + second_rotation_column
            )
            third_q_zero_index = (
                second_head_start
                + (d_model - 1)
                + third_rotation_column
            )
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                (q_all_coordinates != q_pivot_index)
                & (q_all_coordinates != first_q_zero_index)
                & (q_all_coordinates != second_q_zero_index)
                & (q_all_coordinates != third_q_zero_index)
            ]
            q_coordinate_order = torch.cat(
                (
                    q_pivot_index.unsqueeze(0),
                    first_q_zero_index.unsqueeze(0),
                    second_q_zero_index.unsqueeze(0),
                    third_q_zero_index.unsqueeze(0),
                    q_tail_coordinates,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(2),
                self.qkv.weight[: matrix_size - 3],
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
                    matrix_size - 3 : 2 * matrix_size - 4
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
                self.qkv.weight[2 * matrix_size - 4 :],
            )
        )
=======
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(3),
                self.qkv.weight[: matrix_size - 4],
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
                    matrix_size - 4 : 2 * matrix_size - 5
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
                self.qkv.weight[2 * matrix_size - 5 :],
            )
        )
>>>>>>> REPLACE