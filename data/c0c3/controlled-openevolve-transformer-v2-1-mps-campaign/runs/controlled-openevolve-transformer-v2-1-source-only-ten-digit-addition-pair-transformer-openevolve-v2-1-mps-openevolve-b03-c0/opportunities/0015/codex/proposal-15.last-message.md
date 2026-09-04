MECHANISM: Incremental value/output bias redundancy

HYPOTHESIS: Removing a second value-projection bias coordinate will reduce the model from 1,618 to 1,617 parameters while retaining at least 99% accuracy, because attention weights sum to one and the output-projection bias can learn the resulting constant offset.

INTENDED_EDIT: Store 21 learned QKV bias values and reconstruct fixed-zero first-key and first-two-value bias coordinates, preserving the original layer-construction RNG trajectory and all training settings.

EVIDENCE: Removing one key and one value bias coordinate achieved 99.96% accuracy at 1,618 parameters, while removing a second key coordinate failed; extending the successful value-bias reduction by one coordinate is the smallest informative next constraint.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 :],
=======
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[2 * d_model - 1 :],
>>>>>>> REPLACE