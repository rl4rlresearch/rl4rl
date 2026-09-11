MECHANISM: Anchored attention-output common-mode gauge

HYPOTHESIS: Anchoring the final output coordinate of the first attention projection column will reduce the model to 1599 parameters while retaining at least 99% accuracy, because the removed component produces only a feature-common residual shift that downstream LayerNorm removes.

INTENDED_EDIT: Store seven coordinates for the first attention output-projection column, reconstruct its eighth coordinate as zero, and transform the original initialization by subtracting that coordinate from the column.

EVIDENCE: The 1600-parameter design reached 99.93%, and analogous final-coordinate anchoring succeeded for two key rows while the more gradient-coupled third `fc2` centering failed; this tests an untouched downstream common-mode gauge using the successful anchored chart.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class AnchoredOutputLinear(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Consume the same constructor RNG as the original projection layer.
        _ = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.first_column = nn.Parameter(torch.empty(d_model - 1))
        self.rest = nn.Parameter(torch.empty(d_model, d_model - 1))

    def _set_from_full(self, weight: torch.Tensor) -> None:
        first_column = weight[:, 0]
        with torch.no_grad():
            self.first_column.copy_(
                first_column[:-1] - first_column[-1]
            )
            self.rest.copy_(weight[:, 1:])

    def reconstructed_weight(self) -> torch.Tensor:
        first_column = torch.cat(
            (self.first_column, self.first_column.new_zeros(1))
        )
        return torch.cat((first_column.unsqueeze(1), self.rest), dim=1)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
        self.proj = nn.Linear(d_model, d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj.bias = None
=======
        self.qv_bias = nn.Parameter(torch.zeros(2 * d_model - 6))
        self.proj = AnchoredOutputLinear(d_model)
        self.proj_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        y = F.linear(y, self.proj.reconstructed_weight(), proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, AnchoredOutputLinear):
            full_weight = module.first_column.new_empty(
                module.d_model, module.d_model
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module._set_from_full(full_weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE