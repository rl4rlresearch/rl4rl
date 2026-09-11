MECHANISM: Softmax-invariant key-weight centering

HYPOTHESIS: Constraining every key-projection row to the zero-mean input subspace will reduce the model from 1,470 to 1,462 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key shift that cancels exactly in the attention softmax.

INTENDED_EDIT: Store the eight key-projection rows as coefficients in the existing seven-dimensional orthonormal zero-mean basis, reconstruct them during forward passes, and preserve the initialized attention function.

EVIDENCE: The analogous LayerNorm-nullspace parameterization of all 12 MLP input rows retained 99.95% accuracy, while isolated attention-output centering retained 99.93%; unlike the failed MLP-bias constraint, this reduction acts directly through an exact softmax-invariant key shift.

<<<<<<< SEARCH
    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )
=======
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
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

    def gauge_fix_proj(self) -> None:
        with torch.no_grad():
            weight = self.proj.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.proj.weight = nn.Parameter(
                (self.proj_basis.T @ centered).clone()
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_size = d_model * d_model
        k_size = d_model * (d_model - 1)
        q_weight = self.qkv.weight[:q_size].view(d_model, d_model)
        k_coeff = self.qkv.weight[q_size : q_size + k_size].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_weight = self.qkv.weight[q_size + k_size :].view(d_model, d_model)
        q = F.linear(x, q_weight)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
=======
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
>>>>>>> REPLACE