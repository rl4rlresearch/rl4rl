MECHANISM: Final-LayerNorm bias absorption through token-row means

HYPOTHESIS: Restoring the full value bias while quotienting two final-LayerNorm bias degrees of freedom into otherwise hidden token-row means will produce 1,584 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Restore all eight value-bias coordinates, initialize each token embedding row mean-free, and replace the final LayerNorm bias with six learned mean-free coordinates plus a fixed common component that turns token-row means into a direct output-bias channel.

EVIDENCE: The full-value-bias tied-embedding design achieved 99.97% at 1,585 parameters, whereas pruning one value-bias coordinate fell to 97.31%; this keeps the load-bearing value bias and instead removes an exact output-side redundancy while preserving the initial function.

<<<<<<< SEARCH
        # Keep constructor RNG consumption, then remove the softmax-null key bias
        # and prune one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
=======
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), F.pad(v_bias, (0, 1)))
        )
=======
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
=======
class QuotientFinalLayerNorm(nn.LayerNorm):
    """Final LayerNorm with two bias directions absorbed by token-row means."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)

        basis = torch.zeros(normalized_shape, normalized_shape - 2)
        for j in range(normalized_shape - 2):
            scale = math.sqrt((j + 1) * (j + 2))
            basis[: j + 1, j] = 1.0 / scale
            basis[j + 1, j] = -(j + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.register_buffer(
            "fixed_bias",
            torch.full((normalized_shape,), 1.0 / math.sqrt(normalized_shape)),
            persistent=False,
        )
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.fixed_bias + self.bias_basis @ self.bias
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = QuotientFinalLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.init.normal_(full, mean=0.0, std=0.02)
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, MeanFreeTokenEmbedding):
=======
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                embedding.weight.copy_(full.reshape(-1) @ embedding.basis)
        elif isinstance(module, MeanFreeTokenEmbedding):
>>>>>>> REPLACE

<<<<<<< SEARCH
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.weight.copy_(full.reshape(-1) @ module.basis)
        elif isinstance(module, MeanFreePositionEmbedding):
=======
                nn.init.normal_(full, mean=0.0, std=0.02)
                full.sub_(full.mean(dim=1, keepdim=True))
                module.weight.copy_(full.reshape(-1) @ module.basis)
        elif isinstance(module, MeanFreePositionEmbedding):
>>>>>>> REPLACE