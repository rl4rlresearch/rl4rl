MECHANISM: Shared effective query-bias coordinate

HYPOTHESIS: Reconstructing the third query-bias coordinate as the mean of two learned coordinates will produce 1,597 parameters and maintain at least 99% accuracy by preserving three active query offsets.

INTENDED_EDIT: Replace three independent query-bias parameters with two learned parameters whose mean supplies the third effective coordinate.

EVIDENCE: The three-coordinate design achieved 99.97% at 1,598 parameters, while fixing the third coordinate to zero collapsed accuracy to 21.05%; this tests whether the third coordinate must remain active rather than independently parameterized.

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and three query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
=======
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Two learned coordinates generate three active query biases;
        # key/value biases and the remaining query coordinates are fixed at zero.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The learned entries occupy the first three query coordinates. The
        # remaining query coordinates and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
=======
        # Keep the third query coordinate active by sharing the mean of the two
        # independent coordinates, rather than fixing that coordinate at zero.
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.mean().unsqueeze(0))
        )
        bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
>>>>>>> REPLACE