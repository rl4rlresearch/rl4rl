MECHANISM: Single-head reciprocal query–key scale gauge fixing

HYPOTHESIS: Fixing one initialized key coefficient in the first attention head will reduce the model from 1,045 to 1,044 parameters while retaining at least 99% accuracy, because reciprocal scaling of that head’s query map and key map leaves every attention score unchanged.

INTENDED_EDIT: Store the largest initialized first-head key coefficient as a buffer, learn all remaining key coefficients, and reconstruct the complete key map during attention.

EVIDENCE: Fixing the analogous common value-map scaling redundancy while retaining independent head gains achieved 99.95% accuracy at 1,045 parameters; this tests another exact continuous attention gauge one coordinate at a time.

<<<<<<< SEARCH
            self.register_buffer(
                "v_inverse_order", torch.argsort(coordinate_order)
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )
=======
            self.register_buffer(
                "v_inverse_order", torch.argsort(coordinate_order)
            )

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
>>>>>>> REPLACE

<<<<<<< SEARCH
        k_coeff = self.qkv.weight[matrix_size : 2 * matrix_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight[2 * matrix_size :],
            )
        )
=======
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
>>>>>>> REPLACE