MECHANISM: Second-head orthogonal query–key gauge fixing

HYPOTHESIS: Applying the same score-preserving query–key rotation to the second head will reduce the verified model from 1,042 to 1,041 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate two query/key features in the second head to zero one query coefficient, omit that fixed zero from the packed QKV parameter, and reconstruct it during attention.

EVIDENCE: The identical first-head orthogonal gauge removal reduced the model to 1,042 parameters with 99.93% accuracy; the second head has the same independent dot-product-preserving symmetry.

<<<<<<< SEARCH
            q_coeff[:2] = rotation @ q_coeff[:2]
            k_coeff[:2] = rotation @ k_coeff[:2]
            self.qkv.bias[:2].copy_(
                rotation @ self.qkv.bias[:2]
            )
            q_coeff[0, rotation_column] = 0.0
            # Combine the initialized per-head value maps at variance-preserving
=======
            q_coeff[:2] = rotation @ q_coeff[:2]
            k_coeff[:2] = rotation @ k_coeff[:2]
            self.qkv.bias[:2].copy_(
                rotation @ self.qkv.bias[:2]
            )
            q_coeff[0, rotation_column] = 0.0

            # Apply the same independent orthogonal gauge fix to the second
            # head, preserving all of its attention scores exactly.
            second_rotation_column = torch.linalg.vector_norm(
                q_coeff[self.head_dim : self.head_dim + 2], dim=0
            ).argmax()
            a = q_coeff[self.head_dim, second_rotation_column]
            b = q_coeff[self.head_dim + 1, second_rotation_column]
            radius = torch.sqrt(a.square() + b.square())
            second_rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[self.head_dim : self.head_dim + 2] = (
                second_rotation
                @ q_coeff[self.head_dim : self.head_dim + 2]
            )
            k_coeff[self.head_dim : self.head_dim + 2] = (
                second_rotation
                @ k_coeff[self.head_dim : self.head_dim + 2]
            )
            self.qkv.bias[
                self.head_dim : self.head_dim + 2
            ].copy_(
                second_rotation
                @ self.qkv.bias[self.head_dim : self.head_dim + 2]
            )
            q_coeff[
                self.head_dim, second_rotation_column
            ] = 0.0
            # Combine the initialized per-head value maps at variance-preserving
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and the
            # complementary second-head query pivot, while also excluding the
            # first head's orthogonally fixed zero query coordinate.
            q_flat = q_coeff.reshape(-1)
            second_head_start = self.head_dim * (d_model - 1)
            q_pivot_index = second_head_start + q_flat[
                second_head_start:
            ].abs().argmax()
            q_zero_index = rotation_column
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                (q_all_coordinates != q_pivot_index)
                & (q_all_coordinates != q_zero_index)
            ]
            q_coordinate_order = torch.cat(
                (
                    q_pivot_index.unsqueeze(0),
                    q_zero_index.unsqueeze(0),
                    q_tail_coordinates,
                )
            )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[: matrix_size - 2],
            )
        )
=======
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(2),
                self.qkv.weight[: matrix_size - 3],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[
                    matrix_size - 2 : 2 * matrix_size - 3
                ],
=======
                self.qkv.weight[
                    matrix_size - 3 : 2 * matrix_size - 4
                ],
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[2 * matrix_size - 3 :],
=======
                self.qkv.weight[2 * matrix_size - 4 :],
>>>>>>> REPLACE