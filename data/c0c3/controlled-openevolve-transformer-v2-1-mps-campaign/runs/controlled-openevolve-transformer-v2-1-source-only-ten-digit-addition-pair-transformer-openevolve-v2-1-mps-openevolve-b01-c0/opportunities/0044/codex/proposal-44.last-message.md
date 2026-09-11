MECHANISM: Pre-attention LayerNorm bias/projection-bias redundancy

HYPOTHESIS: Removing a third `ln1` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its additive effect can be absorbed by the query bias, is softmax-null for keys, and is representable through the value/output bias path.

INTENDED_EDIT: Store five learned `ln1` bias contrasts instead of six and reconstruct two anchored coordinates before centering the seven active coordinates.

EVIDENCE: Removing a third `ln2` bias degree achieved 99.93% at 1618 parameters, establishing that an additional additive LayerNorm-bias gauge can train successfully; the fourth `ln2` reduction collapsed, so testing the analogous third reduction in the distinct pre-attention LayerNorm is the smallest informative next change.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class ReducedBiasLayerNorm(nn.Module):
=======
        active_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)


class ReducedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE