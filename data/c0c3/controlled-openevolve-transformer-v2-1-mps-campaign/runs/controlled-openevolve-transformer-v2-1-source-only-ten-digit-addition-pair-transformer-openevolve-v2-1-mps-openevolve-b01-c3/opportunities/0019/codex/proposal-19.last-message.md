MECHANISM: Single-coordinate QKV input gauge fixing

HYPOTHESIS: Sharing the verified learned query-bias scalar and fixing one redundant QKV input coefficient will produce a 1,526-parameter model with at least 99% accuracy, because affine-free LayerNorm makes the QKV input mean-zero, so a row’s common input-weight component is functionally invisible.

INTENDED_EDIT: Tie the two query-bias scalars while replacing QKV with a bias-free projection whose first output row fixes one input coefficient to zero; all other projection weights and capacities remain unchanged.

EVIDENCE: The shared-bias 1,527-parameter design reached 99.8%, while fixing its bias amplitude collapsed to 39.69%; this retains the successful learned amplitude and removes a different, exact null direction instead.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class OneRowInputAnchoredLinear(nn.Module):
    """Bias-free linear map with one centered-input gauge coordinate fixed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.first = nn.Linear(in_features - 1, 1, bias=False)
        self.rest = nn.Linear(in_features, out_features - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.first(x[..., :-1]), self.rest(x)), dim=-1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, 1))
=======
        self.qkv = OneRowInputAnchoredLinear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = F.pad(self.q_bias, (0, self.head_dim - 1)).reshape(d_model)
=======
        q_bias = F.pad(
            self.q_bias.expand(self.n_head, 1), (0, self.head_dim - 1)
        ).reshape(d_model)
>>>>>>> REPLACE