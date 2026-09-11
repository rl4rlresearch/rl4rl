MECHANISM: Isolated MLP-output bias gauge fixing

HYPOTHESIS: Constraining only the MLP output bias to the zero-mean feature subspace will reduce the model from 1,470 to 1,469 parameters while retaining at least 99% accuracy, because its removed all-ones component adds only a per-token scalar shift that the final LayerNorm erases under zero dropout.

INTENDED_EDIT: Store the eight-dimensional `fc2` bias as seven orthonormal zero-mean coefficients, reconstruct it during the forward pass, and preserve its initialized observable function.

EVIDENCE: Jointly centering the MLP output weight and bias caused the 1,457-parameter model to fail at 84.22%, while isolated attention-output centering reached 99.93% at 1,470 parameters; isolating the single MLP bias gauge tests whether the prior failure came from constraining the output weights rather than the exact bias redundancy.

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

    def gauge_fix_fc2_bias(self) -> None:
        with torch.no_grad():
            bias = self.fc2.bias
            centered = bias - bias.mean()
            self.fc2.bias = nn.Parameter(
                (self.fc1_basis @ centered).clone()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight = self.fc1.weight @ self.fc1_basis
        hidden = F.linear(x, fc1_weight, self.fc1.bias)
        fc2_bias = self.fc1_basis.T @ self.fc2.bias
        output = F.linear(F.gelu(hidden), self.fc2.weight, fc2_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
=======
        for block in self.blocks:
            block.attn.gauge_fix_proj()
            block.mlp.gauge_fix_fc1()
            block.mlp.gauge_fix_fc2_bias()
>>>>>>> REPLACE