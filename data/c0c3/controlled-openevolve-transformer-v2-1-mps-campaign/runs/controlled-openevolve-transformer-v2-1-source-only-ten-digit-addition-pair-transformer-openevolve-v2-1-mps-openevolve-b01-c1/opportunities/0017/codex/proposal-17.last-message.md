MECHANISM: LayerNorm-quotient residual output weights

HYPOTHESIS: Constraining both residual-output weight matrices to the seven-dimensional mean-free output subspace will reduce parameters from 1,615 to 1,595 while retaining at least 99% accuracy, because the removed input-dependent all-ones components pass unchanged through residual connections and are canceled by downstream LayerNorms.

INTENDED_EDIT: Represent the attention projection and MLP `fc2` weights with orthonormal mean-free coordinates, reconstruct full weights during forward passes, and preserve full-width initialization RNG consumption.

EVIDENCE: Mean-free quotients already reduced the positional embeddings and both residual-output biases while retaining 99.64%–99.98% accuracy; applying the same exact LayerNorm-invariant quotient independently to every weight column removes 20 redundant parameters without narrowing observable output capacity.

<<<<<<< SEARCH
class MeanFreeResidualLinear(nn.Linear):
    """Linear output bias modulo the constant direction removed by LayerNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-1].clone())

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, self.weight, full_bias)
=======
class MeanFreeResidualLinear(nn.Linear):
    """Residual linear map modulo output components removed by LayerNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        full_bias = self.bias.detach()

        basis = torch.zeros(out_features, out_features - 1)
        for j in range(out_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)

        self.weight = nn.Parameter((basis.transpose(0, 1) @ full_weight).clone())
        self.bias = nn.Parameter((full_bias @ basis).clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = self.bias_basis @ self.weight
        full_bias = self.bias @ self.bias_basis.transpose(0, 1)
        return F.linear(x, full_weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, MeanFreeResidualLinear):
            # Draw the original full output matrix so subsequent initialization
            # keeps the same RNG sequence, then retain its observable component.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.bias_basis.transpose(0, 1) @ full)
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE