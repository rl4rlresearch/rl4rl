MECHANISM: Common value-scale gauge fixing

HYPOTHESIS: Removing one redundant shared-value scale coordinate will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because both independent head gains remain learned and can absorb any common rescaling of the shared value map.

INTENDED_EDIT: Fix the largest initialized shared-value coefficient as a buffer, learn the other 27 coefficients, and reconstruct the full value map during attention while preserving both head-specific gains.

EVIDENCE: Disjoint routing with two learned gains achieved 99.96% at 1,046 parameters, whereas tying those gains reduced accuracy to 95.25%; this patch preserves their independence and removes only the exact common scaling redundancy between the shared value map and both gains.

<<<<<<< SEARCH
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
            v_coeff = v_weight @ self.proj_basis

            # A common rescaling of the shared value map can be absorbed by
            # both independent head gains. Fix its largest initialized
            # coordinate and learn every remaining value coordinate directly.
            v_flat = v_coeff.reshape(-1)
            pivot_index = v_flat.abs().argmax()
            all_coordinates = torch.arange(
                v_flat.numel(), device=v_flat.device
            )
            tail_coordinates = all_coordinates[
                all_coordinates != pivot_index
            ]
            coordinate_order = torch.cat(
                (pivot_index.unsqueeze(0), tail_coordinates)
            )
            self.register_buffer(
                "v_pivot", v_flat[pivot_index].detach().clone()
            )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
=======
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight[2 * matrix_size :],
            )
        )
        v_coeff = v_flat[self.v_inverse_order].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
>>>>>>> REPLACE