MECHANISM: LayerNorm–projection scale gauge fixing

HYPOTHESIS: Fixing one first-LayerNorm scale coordinate to one will reduce the model from 1608 to 1607 parameters while retaining at least 99% accuracy, because that scale can be absorbed exactly into the corresponding dense QKV projection column.

INTENDED_EDIT: Replace the first LayerNorm with an equivalent implementation containing seven learned scales and one fixed unit scale, preserving all verified attention biases and dense projection weights.

EVIDENCE: The 1608-parameter model achieved 99.88%, while removing a query-bias coordinate or sparsifying a key row failed; this targets a multiplicative LayerNorm/projection redundancy without removing those capacities or changing the initial function.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class ScaleFixedLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = ScaleFixedLayerNorm(cfg.d_model)
>>>>>>> REPLACE