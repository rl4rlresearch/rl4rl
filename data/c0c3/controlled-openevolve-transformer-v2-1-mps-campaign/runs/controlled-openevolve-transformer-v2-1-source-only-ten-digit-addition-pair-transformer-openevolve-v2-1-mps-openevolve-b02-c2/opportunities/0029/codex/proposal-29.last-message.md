MECHANISM: Pre-MLP LayerNorm scale gauge fixing

HYPOTHESIS: Fixing one `ln2` scale coordinate at its initialized value of one will reduce the model to 1,629 parameters while retaining at least 99% accuracy, because the unrestricted corresponding `fc1` weight column can learn the same multiplicative transformation.

INTENDED_EDIT: Replace `ln2` with an equivalent LayerNorm containing seven learned scale coordinates and all eight learned biases, preserving the baseline initialization and leaving the qualified attention parameterization unchanged.

EVIDENCE: The current attention layout achieved 99.81% at 1,630 parameters, while tying an attention output bias fell to 70.57%; this motivates preserving attention and testing a distinct multiplicative gauge. The failed `ln2` beta omission modified an additive pathway, whereas this patch retains every LayerNorm bias and removes only a scale redundant with `fc1`.

<<<<<<< SEARCH
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
=======
        return F.linear(x, self.weight, full_bias)


class CompactLayerNorm(nn.Module):
    """LayerNorm with its first scale fixed at the initialized value one."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = nn.Parameter(layer_norm.weight[1:].detach().clone())
        self.bias = layer_norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat((self.weight.new_ones(1), self.weight))
        return F.layer_norm(
            x,
            self.normalized_shape,
            full_weight,
            self.bias,
            self.eps,
        )


class MLP(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3) and value biases in a 4:3 per-head layout.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Preserve every baseline initialization draw before removing seven
        # key biases (4:3), value biases in a 4:3 per-head layout, and one
        # pre-MLP LayerNorm scale coordinate.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.ln2 = CompactLayerNorm(block.ln2)
>>>>>>> REPLACE