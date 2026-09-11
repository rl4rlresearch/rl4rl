MECHANISM: Third fixed value-feature intercept

HYPOTHESIS: A 805-parameter transformer will retain at least 99% accuracy because fixing the second value-feature bias at zero preserves all four gated nonlinear features, every input-dependent value and gate weight, and the qualified lexical and attention paths.

INTENDED_EDIT: Extend the qualified 806-parameter MLP bias constraint by learning only its first input-projection bias and fixing the remaining seven intercepts at zero.

EVIDENCE: The 806-parameter design achieved 99.96% after fixing two value-feature and all four gate intercepts; its strong margin motivates the narrowest adjacent one-parameter reduction within the repeatedly successful bias-fixing mechanism.

<<<<<<< SEARCH
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final two output biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 3:
            raise ValueError("biased output must have at least three features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 2))
            if bias
            else None
        )
=======
class QuotientInputLinear(nn.Module):
    """Zero-mean-input linear map with its final seven biases fixed."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features < 2:
            raise ValueError("in_features must be at least two")
        if bias and out_features < 8:
            raise ValueError("biased output must have at least eight features")

        self.coeff = nn.Linear(in_features - 1, out_features, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 7))
            if bias
            else None
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = None if self.bias is None else F.pad(self.bias, (0, 2))
        return F.linear(quotient_x, self.coeff.weight, bias)
=======
        bias = None if self.bias is None else F.pad(self.bias, (0, 7))
        return F.linear(quotient_x, self.coeff.weight, bias)
>>>>>>> REPLACE