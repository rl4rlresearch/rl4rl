MECHANISM: LayerNorm-induced key-weight gauge quotient

HYPOTHESIS: Removing one common-direction coordinate from a key-projection row will reduce the model from 1,584 to 1,583 parameters while retaining at least 99% accuracy, because LayerNorm initially produces mean-free features and the removed component changes every key by the same position-independent scalar, which causal-attention softmax cancels.

INTENDED_EDIT: Represent one key-projection row in a seven-dimensional orthonormal mean-free basis, reconstruct the full QKV matrix during attention, and preserve the original full-width initialization draw.

EVIDENCE: Eliminating all eight softmax-null key-bias coordinates retained 99.94% accuracy, while tying value and output biases failed at 62.77%; this motivates extending the proven key-side softmax invariance instead of coupling optimization-sensitive value pathways.

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class CausalSelfAttention(nn.Module):
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embedding.full_weight())


class OneQuotientKeyLinear(nn.Linear):
    """QKV projection with one softmax-null key-row direction removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.key_row = out_features // 3

        basis = torch.zeros(in_features, in_features - 1)
        for j in range(in_features - 1):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("key_basis", basis, persistent=False)

        full_weight = self.weight.detach()
        self.weight = nn.Parameter(
            self.coordinates_from_full(full_weight).clone()
        )

    def coordinates_from_full(self, full_weight: torch.Tensor) -> torch.Tensor:
        prefix = full_weight[: self.key_row].reshape(-1)
        key = full_weight[self.key_row] @ self.key_basis
        suffix = full_weight[self.key_row + 1 :].reshape(-1)
        return torch.cat((prefix, key, suffix))

    def full_weight(self) -> torch.Tensor:
        prefix_size = self.key_row * self.in_features
        key_end = prefix_size + self.in_features - 1
        prefix = self.weight[:prefix_size].view(self.key_row, self.in_features)
        key = self.weight[prefix_size:key_end] @ self.key_basis.transpose(0, 1)
        suffix = self.weight[key_end:].view(
            self.out_features - self.key_row - 1, self.in_features
        )
        return torch.cat((prefix, key.unsqueeze(0), suffix), dim=0)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
=======
        self.qkv = OneQuotientKeyLinear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.full_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, TiedMeanFreeOutput):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, OneQuotientKeyLinear):
            # Draw the original full QKV matrix, then quotient one key row by
            # the common input direction that only shifts all key logits.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(module.coordinates_from_full(full))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, TiedMeanFreeOutput):
>>>>>>> REPLACE