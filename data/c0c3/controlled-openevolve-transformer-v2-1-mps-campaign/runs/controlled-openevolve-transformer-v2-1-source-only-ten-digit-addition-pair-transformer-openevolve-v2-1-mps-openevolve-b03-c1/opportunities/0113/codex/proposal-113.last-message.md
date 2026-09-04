MECHANISM: Independent feature-wise query–key scale gauge fixing

HYPOTHESIS: Fixing one second-head key coefficient in a feature distinct from the existing second-head query pivot will reduce the model from 1,041 to 1,040 parameters while retaining at least 99% accuracy, because reciprocal scaling of that feature’s query row and key row preserves every attention score.

INTENDED_EDIT: Select the largest initialized eligible second-head key coefficient as a fixed buffer, exclude it from the packed QKV parameter, and reconstruct both key pivots during attention.

EVIDENCE: Reciprocal query–key gauge fixes reached 1,043 parameters with 99.98% accuracy, and subsequent score-preserving orthogonal fixes reached 1,041 parameters with 99.90%. This uses another exact query–key symmetry while avoiding the additional rotation logic associated with the recent timeouts.

<<<<<<< SEARCH
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and the
            # complementary second-head query pivot, while excluding both
            # orthogonally fixed zero query coordinates.
=======
            # Each query/key feature pair has an independent reciprocal
            # scaling symmetry. Retain the first-head key pivot and the
            # complementary second-head query pivot, exclude both orthogonally
            # fixed zero query coordinates, and below fix an additional key
            # coordinate from a different second-head feature.
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            k_flat = k_coeff.reshape(-1)
            first_head_size = self.head_dim * (d_model - 1)
            k_pivot_index = k_flat[:first_head_size].abs().argmax()
            k_all_coordinates = torch.arange(
                k_flat.numel(), device=k_flat.device
            )

            # The existing second-head query pivot fixes only its own feature
            # scale. Fix the best-conditioned key coefficient from another
            # feature, whose reciprocal query/key scale remains independent.
            second_head_coordinates = k_all_coordinates[first_head_size:]
            second_head_rows = (
                second_head_coordinates // (d_model - 1)
            )
            q_pivot_row = q_pivot_index // (d_model - 1)
            eligible_second_head_coordinates = second_head_coordinates[
                second_head_rows != q_pivot_row
            ]
            second_k_pivot_index = eligible_second_head_coordinates[
                k_flat[eligible_second_head_coordinates].abs().argmax()
            ]
            k_tail_coordinates = k_all_coordinates[
                (k_all_coordinates != k_pivot_index)
                & (k_all_coordinates != second_k_pivot_index)
            ]
            k_coordinate_order = torch.cat(
                (
                    k_pivot_index.unsqueeze(0),
                    second_k_pivot_index.unsqueeze(0),
                    k_tail_coordinates,
                )
            )
            self.register_buffer(
                "k_pivot", k_flat[k_pivot_index].detach().clone()
            )
            self.register_buffer(
                "second_k_pivot",
                k_flat[second_k_pivot_index].detach().clone(),
            )
            self.register_buffer(
                "k_inverse_order", torch.argsort(k_coordinate_order)
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
                self.second_k_pivot.unsqueeze(0),
                self.qkv.weight[
                    matrix_size - 3 : 2 * matrix_size - 5
                ],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.weight[2 * matrix_size - 4 :],
=======
                self.qkv.weight[2 * matrix_size - 5 :],
>>>>>>> REPLACE