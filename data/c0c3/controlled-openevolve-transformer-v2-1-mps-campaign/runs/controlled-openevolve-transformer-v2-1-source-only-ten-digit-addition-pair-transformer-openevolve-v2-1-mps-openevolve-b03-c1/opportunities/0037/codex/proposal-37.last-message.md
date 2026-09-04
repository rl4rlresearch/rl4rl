MECHANISM: LayerNorm-nullspace value-weight centering

HYPOTHESIS: Constraining every value-projection row to the zero-mean input subspace will reduce the model from 1,462 to 1,454 learned parameters while retaining at least 99% accuracy, because the removed component is position-independent after `ln1` and is exactly representable by the existing value bias.

INTENDED_EDIT: Store each 8-by-8 value projection as 8-by-7 coefficients in the existing orthonormal zero-mean basis and reconstruct its centered weight during every forward pass.

EVIDENCE: The analogous key-weight centering reached 100% accuracy at 1,462 parameters, and MLP input-weight centering reached 99.95%; value centering uses the same LayerNorm-null direction while retaining a full value bias to represent its constant contribution.

<<<<<<< SEARCH
            v_weight = weight[2 * d_model :]
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_weight.reshape(-1),
                        k_coeff.reshape(-1),
                        v_weight.reshape(-1),
                    )
                ).clone()
            )
=======
            v_weight = weight[2 * d_model :]
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis
            v_weight = v_weight - v_weight.mean(dim=1, keepdim=True)
            v_coeff = v_weight @ self.proj_basis
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_weight.reshape(-1),
                        k_coeff.reshape(-1),
                        v_coeff.reshape(-1),
                    )
                ).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        k_weight = k_coeff @ self.proj_basis.T
        v_weight = self.qkv.weight[q_size + k_size :].view(d_model, d_model)
        q = F.linear(x, q_weight)
=======
        k_weight = k_coeff @ self.proj_basis.T
        v_coeff = self.qkv.weight[q_size + k_size :].view(
            d_model, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight)
>>>>>>> REPLACE