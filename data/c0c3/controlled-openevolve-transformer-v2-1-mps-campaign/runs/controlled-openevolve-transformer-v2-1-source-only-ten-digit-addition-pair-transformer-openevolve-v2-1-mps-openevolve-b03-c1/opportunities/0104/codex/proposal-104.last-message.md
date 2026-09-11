MECHANISM: Second-head reciprocal query–key scale gauge fixing

HYPOTHESIS: Fixing one initialized key coefficient in the second attention head will reduce the verified model from 1,044 to 1,043 parameters while retaining at least 99% accuracy, because each head has an independent reciprocal query–key scaling symmetry and fixing the first head’s analogous coefficient achieved 99.99% accuracy.

INTENDED_EDIT: Fix the largest initialized key coefficient in each attention head as a buffer, learn all remaining key coefficients, and reconstruct the complete key map during attention.

EVIDENCE: The current 1,044-parameter design fixed one first-head key coefficient and achieved 99.99% accuracy, directly validating this exact gauge removal; applying the same independent symmetry to the second head is the smallest evidence-backed next change.

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
=======
            # Each head has an independent reciprocal query/key scaling gauge.
            # Fix the largest initialized key coefficient in both heads and
            # learn every remaining coordinate.
            k_flat = k_coeff.reshape(-1)
            first_head_size = self.head_dim * (d_model - 1)
            first_k_pivot = k_flat[:first_head_size].abs().argmax()
            second_k_pivot = first_head_size + k_flat[
                first_head_size:
            ].abs().argmax()
            k_pivot_indices = torch.stack(
                (first_k_pivot, second_k_pivot)
            )
            k_all_coordinates = torch.arange(
                k_flat.numel(), device=k_flat.device
            )
            k_tail_coordinates = k_all_coordinates[
                (
                    k_all_coordinates[:, None]
                    != k_pivot_indices[None, :]
                ).all(dim=1)
            ]
            k_coordinate_order = torch.cat(
                (k_pivot_indices, k_tail_coordinates)
            )
            self.register_buffer(
                "k_pivot", k_flat[k_pivot_indices].detach().clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[matrix_size : 2 * matrix_size - 1],
            )
        )
=======
        k_flat = torch.cat(
            (
                self.k_pivot,
                self.qkv.weight[matrix_size : 2 * matrix_size - 2],
            )
        )
>>>>>>> REPLACE