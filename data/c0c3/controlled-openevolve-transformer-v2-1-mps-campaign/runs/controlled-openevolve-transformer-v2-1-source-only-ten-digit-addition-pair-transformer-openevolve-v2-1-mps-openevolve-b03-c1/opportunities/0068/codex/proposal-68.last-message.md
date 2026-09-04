MECHANISM: Orthogonal value/output latent-gauge fixing

HYPOTHESIS: Fixing one value-projection coefficient through an orthogonal value/output basis rotation will reduce the model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because it preserves the initialized function and all learned attention-score routing coordinates.

INTENDED_EDIT: Rotate the shared value representation, compensate both attention-output projection blocks, omit the resulting fixed-zero value coefficient, and reconstruct it during forward passes.

EVIDENCE: The 1,119-parameter design achieved 99.53%, while pruning another routing bias fell to 73.42% and altering the query/key factorization reached only 90.12%; this motivates preserving those coordinates and removing an exact degree of freedom from the distinct consecutive linear value/output factorization.

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

            # The shared value features and each head's corresponding output
            # projection block have an exact orthogonal change-of-basis gauge.
            # Rotate the first value coefficient to zero and compensate both
            # output blocks before omitting that fixed coordinate.
            a = v_coeff[0, 0]
            b = v_coeff[1, 0]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.eye(
                self.head_dim,
                device=v_coeff.device,
                dtype=v_coeff.dtype,
            )
            rotation[0, 0] = b / radius
            rotation[0, 1] = -a / radius
            rotation[1, 0] = a / radius
            rotation[1, 1] = b / radius
            v_coeff = rotation @ v_coeff
            v_coeff[0, 0] = 0.0

            proj_weight = self.proj.weight.view(
                d_model, self.n_head, self.head_dim
            )
            self.proj.weight.copy_(
                (proj_weight @ rotation.T).reshape(d_model, d_model)
            )

            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_coeff.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1)[1:],
                    )
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        v_coeff = self.qkv.weight[2 * matrix_size :].view(
            self.head_dim, d_model - 1
        )
=======
        v_coeff = torch.cat(
            (
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[2 * matrix_size :],
            )
        ).view(self.head_dim, d_model - 1)
>>>>>>> REPLACE