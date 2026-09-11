MECHANISM: Direct LayerNorm-2 bias absorption gauge

HYPOTHESIS: Fixing LayerNorm-2 bias feature 7 without adding another virtual optimizer coordinate will reduce the model from 1604 to 1603 parameters, finish within the time limit, and retain at least 99% accuracy because its constant downstream effect is absorbable by `fc1.bias`.

INTENDED_EDIT: Retain only feature 5 of the LayerNorm-2 bias and reconstruct feature 7 as zero, leaving the existing optimizer machinery unchanged.

EVIDENCE: The 1604-parameter model reached 99.92% while LayerNorm-2 bias features 0, 2, 3, and 4 were already fixed without virtual-coordinate optimization; the feature-7 trial timed out only after extending that machinery, motivating a direct omission of the same absorbable coordinate.

<<<<<<< SEARCH
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and bias 0, 1, 2, 3, 4, 6 absorbed downstream."""
=======
class GaugeFixedLayerNormScale(nn.Module):
    """LayerNorm with scale 0, 1, 4 and only bias feature 5 learned."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 6))
=======
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
            )
        )
=======
        full_bias = torch.cat(
            (
                self.bias.new_zeros(5),
                self.bias,
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE