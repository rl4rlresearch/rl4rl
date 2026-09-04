MECHANISM: LayerNorm-null query-weight quotient

HYPOTHESIS: Fixing one query projection input coefficient at zero will reduce the model from 1,520 to 1,519 parameters while retaining at least 99% accuracy, because causal attention receives mean-zero normalized activations and the removed coefficient can be absorbed into the other coefficients of that query row.

INTENDED_EDIT: Replace the QKV projection with a specialized linear layer that omits the final input coefficient of its first query row, reconstructs it as zero, and canonicalizes initialization by subtracting the omitted coefficient from the retained row entries.

EVIDENCE: The verified 1,520-parameter design already uses twelve analogous LayerNorm-input weight quotients in the MLP while reaching 99.83%; this tests that successful mechanism in one attention query row instead of extending the value-bias reduction whose third coordinate fell to 94.05%.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class SinglePrunedQKVLinear(nn.Linear):
    """QKV map with one LayerNorm-null query coefficient removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        full_weight = self.weight.detach()
        self.first_query_row = nn.Parameter(full_weight[0, :-1].clone())
        self.weight = nn.Parameter(full_weight[1:].clone())

    def reconstructed_weight(self) -> torch.Tensor:
        first_query_row = F.pad(self.first_query_row, (0, 1)).unsqueeze(0)
        return torch.cat((first_query_row, self.weight), dim=0)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = SinglePrunedQKVLinear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.reconstructed_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
=======
        elif isinstance(module, SinglePrunedQKVLinear):
            # Draw the original full-width projection, then canonicalize the
            # first query row using the zero-sum LayerNorm input.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                module.first_query_row.copy_(
                    full[0, :-1] - full[0, -1]
                )
                module.weight.copy_(full[1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, DistributedTwelvePrunedInputWeightLinear):
>>>>>>> REPLACE