MECHANISM: Single-coordinate orthogonal query/key gauge fixing

HYPOTHESIS: Fixing one query-weight coordinate through a score-preserving orthogonal query/key rotation will reduce the model from 1,535 to 1,534 parameters while retaining at least 99% accuracy, avoiding the optimization shock caused by fixing all 32 attention-basis gauges simultaneously.

INTENDED_EDIT: Replace the QKV projection with a one-parameter-smaller projection whose first query coordinate is fixed at zero, and orthogonally canonicalize the corresponding query/key rows during fresh initialization while preserving initial attention scores.

EVIDENCE: The 1,535-parameter design reached 99.87%, whereas removing all 32 query/key basis-gauge parameters at once collapsed to 23.28% despite preserving score functions. Fixing only one gauge coordinate is the smallest test of whether that failure resulted from aggressive simultaneous reparameterization.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
=======
class OneGaugeQKVLinear(nn.Linear):
    """QKV projection with one orthogonal query/key basis gauge fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        full_weight = self.weight.detach()
        self.first_query_row = nn.Parameter(full_weight[0, 1:].clone())
        self.weight = nn.Parameter(full_weight[1:].clone())

    def full_weight(self) -> torch.Tensor:
        first_query_row = F.pad(self.first_query_row, (1, 0)).unsqueeze(0)
        return torch.cat((first_query_row, self.weight), dim=0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = OneGaugeQKVLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight) + full_bias
=======
        qkv = F.linear(x, self.qkv.full_weight()) + full_bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, DistributedElevenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
=======
        elif isinstance(module, OneGaugeQKVLinear):
            # Draw the original full QKV matrix, then orthogonally rotate the
            # first two coordinates of head zero in both query and key space.
            # This preserves every initial attention score while making the
            # first query row's first input weight exactly zero.
            with torch.no_grad():
                full = torch.empty(
                    module.out_features,
                    module.in_features,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                nn.init.normal_(full, mean=0.0, std=0.02)
                a = full[0, 0].clone()
                b = full[1, 0].clone()
                radius = torch.sqrt(a.square() + b.square())
                q0 = (b * full[0] - a * full[1]) / radius
                q1 = (a * full[0] + b * full[1]) / radius
                key_offset = module.in_features
                k0 = (
                    b * full[key_offset] - a * full[key_offset + 1]
                ) / radius
                k1 = (
                    a * full[key_offset] + b * full[key_offset + 1]
                ) / radius
                full[1].copy_(q1)
                full[key_offset].copy_(k0)
                full[key_offset + 1].copy_(k1)
                module.first_query_row.copy_(q0[1:])
                module.weight.copy_(full[1:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, DistributedElevenPrunedInputWeightLinear):
            # Preserve the original full-width draw. At the initial unit
>>>>>>> REPLACE