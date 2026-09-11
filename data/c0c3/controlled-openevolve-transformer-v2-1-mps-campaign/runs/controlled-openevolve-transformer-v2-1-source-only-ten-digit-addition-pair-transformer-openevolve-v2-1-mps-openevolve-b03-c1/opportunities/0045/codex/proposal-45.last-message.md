MECHANISM: Final-LayerNorm-null MLP output-weight centering

HYPOTHESIS: Constraining only the MLP output weight to the seven-dimensional zero-mean residual subspace will reduce parameters from 1,221 to 1,209 while retaining at least 99% accuracy, because each removed all-ones output component is erased by the final LayerNorm while the full eight-dimensional MLP output bias remains unchanged.

INTENDED_EDIT: Store `fc2.weight` as 7-by-12 coefficients, reconstruct its centered 8-by-12 weight during forward passes, and preserve its initialized observable function.

EVIDENCE: Jointly centering the MLP output weight and bias failed at 84.22%, but centering the bias alone also failed at 79.56%; therefore the weight-only constraint is the missing isolation experiment. The same orthonormal output-subspace parameterization succeeded for the attention projection at 99.93%.

<<<<<<< SEARCH
    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
    def gauge_fix_fc1(self) -> None:
        with torch.no_grad():
            weight = self.fc1.weight
            centered = weight - weight.mean(dim=1, keepdim=True)
            self.fc1.weight = nn.Parameter(
                (centered @ self.fc1_basis.T).clone()
            )

    def gauge_fix_fc2(self) -> None:
        with torch.no_grad():
            weight = self.fc2.weight
            centered = weight - weight.mean(dim=0, keepdim=True)
            self.fc2.weight = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_weight = self.fc1_basis.T @ self.fc2.weight
        output = F.linear(F.gelu(hidden), fc2_weight, self.fc2.bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
=======
        for block in self.blocks:
            block.attn.gauge_fix_qkv()
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
            block.mlp.gauge_fix_fc2()
>>>>>>> REPLACE