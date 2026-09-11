MECHANISM: First-head orthogonal query–key gauge fixing

HYPOTHESIS: Removing one query coefficient through a score-preserving joint rotation of the first head’s query and key feature spaces will reduce the model from 1,043 to 1,042 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Apply an initialized orthogonal rotation that zeros one first-head query coefficient, omit that fixed zero from the packed trainable QKV parameter, and reconstruct it during attention.

EVIDENCE: Independent reciprocal query–key gauge fixes reduced the model to 1,043 parameters with 99.98% accuracy; this uses another exact query–key symmetry while preserving both head routes and every attention score.

<<<<<<< SEARCH
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            # Combine the initialized per-head value maps at variance-preserving
=======
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis

            # A joint orthogonal rotation of a head's query and key features
            # leaves every dot-product attention score unchanged. Rotate the
            # first two features of the first head so one query coefficient is
            # exactly zero, then omit that fixed gauge coordinate below.
            rotation_column = torch.linalg.vector_norm(
                q_coeff[:2], dim=0
            ).argmax()
            a = q_coeff[0, rotation_column]
            b = q_coeff[1, rotation_column]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[:2] = rotation @ q_coeff[:2]
            k_coeff[:2] = rotation @ k_coeff[:2]
            self.qkv.bias[:2].copy_(
                rotation @ self.qkv.bias[:2]
            )
            q_coeff[0, rotation_column] = 0.0
            # Combine the initialized per-head value maps at variance-preserving
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
            self.register_buffer(
                "q_pivot", q_flat[q_pivot_index].detach().clone()
            )
            self.register_buffer(
                "q_inverse_order", torch.argsort(q_coordinate_order)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight[: matrix_size - 1],
            )
        )
=======
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[: matrix_size - 2],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[
                    matrix_size - 1 : 2 * matrix_size - 2
                ],
=======
                self.qkv.weight[
                    matrix_size - 2 : 2 * matrix_size - 3
                ],
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[2 * matrix_size - 2 :],
=======
                self.qkv.weight[2 * matrix_size - 3 :],
>>>>>>> REPLACE