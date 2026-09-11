MECHANISM: Additional first-head orthogonal query–key gauge fixing

HYPOTHESIS: Eliminating a second first-head query coefficient through a score-preserving query–key rotation will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the second and third features of the first attention head to zero one additional query coefficient, exclude that fixed zero from the packed QKV parameter, and reconstruct it during attention.

EVIDENCE: The first- and second-head orthogonal gauge removals preserved all attention scores and achieved 99.93% at 1,042 parameters and 99.90% at 1,041 parameters respectively, directly supporting another independent orthogonal gauge removal.

<<<<<<< SEARCH
            q_coeff[0, rotation_column] = 0.0

            # Apply the same independent orthogonal gauge fix to the second
            # head, preserving all of its attention scores exactly.
=======
            q_coeff[0, rotation_column] = 0.0

            # The first row is now fixed without constraining rotations among
            # the next two features. Use that residual orthogonal freedom to
            # eliminate one more first-head query coefficient while preserving
            # every attention score and the existing zero.
            residual_rotation_column = torch.linalg.vector_norm(
                q_coeff[1:3], dim=0
            ).argmax()
            a = q_coeff[1, residual_rotation_column]
            b = q_coeff[2, residual_rotation_column]
            radius = torch.sqrt(a.square() + b.square())
            residual_rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[1:3] = residual_rotation @ q_coeff[1:3]
            k_coeff[1:3] = residual_rotation @ k_coeff[1:3]
            self.qkv.bias[1:3].copy_(
                residual_rotation @ self.qkv.bias[1:3]
            )
            q_coeff[1, residual_rotation_column] = 0.0

            # Apply the same independent orthogonal gauge fix to the second
            # head, preserving all of its attention scores exactly.
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
            residual_q_zero_index = (
                d_model - 1 + residual_rotation_column
            )
            second_q_zero_index = (
                second_head_start + second_rotation_column
            )
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                (q_all_coordinates != q_pivot_index)
                & (q_all_coordinates != first_q_zero_index)
                & (q_all_coordinates != residual_q_zero_index)
                & (q_all_coordinates != second_q_zero_index)
            ]
            q_coordinate_order = torch.cat(
                (
                    q_pivot_index.unsqueeze(0),
                    first_q_zero_index.unsqueeze(0),
                    residual_q_zero_index.unsqueeze(0),
                    second_q_zero_index.unsqueeze(0),
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
=======
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(3),
                self.qkv.weight[: matrix_size - 4],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[
                    matrix_size - 3 : 2 * matrix_size - 4
                ],
            )
        )
=======
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[
                    matrix_size - 4 : 2 * matrix_size - 5
                ],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[2 * matrix_size - 4 :],
=======
                self.qkv.weight[2 * matrix_size - 5 :],
>>>>>>> REPLACE