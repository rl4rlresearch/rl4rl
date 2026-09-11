MECHANISM: Single-coordinate redundant key-bias removal

HYPOTHESIS: Removing one key-bias coordinate will reduce the model from 1,644 to 1,643 learned parameters while retaining at least 99% accuracy because that coordinate contributes the same additive attention-logit shift at every attended position and therefore cancels exactly under softmax.

INTENDED_EDIT: Replace the QKV projection with a compatible linear subclass whose bias omits one key coordinate and reconstructs it as zero during the forward pass, while preserving parameter ordering, weight initialization, and all other architecture and training settings.

EVIDENCE: The original 1,644-parameter model reached 99.96%, while removing all eight redundant key-bias coordinates reached 88.89%; testing only one coordinate is the smallest possible perturbation and isolates whether the larger simultaneous reparameterization caused the optimization failure.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
=======
class SingleReducedKeyBiasLinear(nn.Linear):
    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model, bias=False)
        self.d_model = d_model
        self.bias = nn.Parameter(torch.zeros(3 * d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omitted = self.bias.new_zeros(1)
        split = 2 * self.d_model - 1
        full_bias = torch.cat((self.bias[:split], omitted, self.bias[split:]))
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, max_seq_len: int):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.qkv = SingleReducedKeyBiasLinear(d_model)
>>>>>>> REPLACE