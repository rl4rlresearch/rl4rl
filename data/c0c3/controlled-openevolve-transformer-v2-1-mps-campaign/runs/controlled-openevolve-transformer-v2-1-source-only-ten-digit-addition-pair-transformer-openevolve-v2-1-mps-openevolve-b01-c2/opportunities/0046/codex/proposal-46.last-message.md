MECHANISM: Residual-stream common-shift bias gauge

HYPOTHESIS: Combining the verified four-row balanced QKV gauge with exact common-shift gauges on both residual-branch output biases will reduce the model from 1,583 to 1,581 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce query-row gauges 0, 1, 4, and 5, then replace the attention projection and MLP output linears with learned linears that omit one redundant bias coordinate each.

EVIDENCE: Reference Design 3 achieved 99.98% accuracy with 1,583 parameters using four balanced query-row gauges. A common shift in either residual-branch output is preserved through pre-norm residual blocks and canceled by the final LayerNorm, making one bias direction per branch functionally redundant.

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1)
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        # Balance two query-row gauges across each attention head.
        self.gauged_rows = (0, 1, head_dim, head_dim + 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, fused_bias)


class ResidualGaugedLinear(nn.Linear):
    """Linear whose output bias fixes the residual common-shift gauge."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.new_empty(out_features - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.linear(x, self.weight, bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias while gauge-fixing two normalized-input QKV rows.
        self.qkv = LayerNormGaugedQKV(d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        # Retain query bias while gauge-fixing two query rows in each head.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
        self.proj = ResidualGaugedLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = nn.Linear(d_ff, d_model)
=======
        self.fc2 = ResidualGaugedLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves both selected row functions.
=======
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves all selected row functions.
>>>>>>> REPLACE