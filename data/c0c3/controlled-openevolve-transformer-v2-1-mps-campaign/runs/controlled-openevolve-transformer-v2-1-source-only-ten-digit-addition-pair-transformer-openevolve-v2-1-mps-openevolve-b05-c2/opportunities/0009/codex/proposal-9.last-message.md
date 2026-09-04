MECHANISM: LayerNorm-offset removal with one-coordinate value-bias anchoring

HYPOTHESIS: Combining the qualified seven-coordinate value bias with the qualified bias-free `ln2` will reduce the model from 1583 to 1582 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the 16-parameter query/value bias tensor with an 8-parameter query bias and 7-parameter value bias, reconstructing the final value coordinate as zero.

EVIDENCE: Seven value-bias coordinates achieved 99.94% accuracy at 1590 parameters, and independently removing `ln2` bias achieved 99.96% at 1583; the failures at six value coordinates and without `ln1` bias favor this conservative combination of qualified reductions.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
=======
        value_bias = F.pad(self.v_bias, (0, 1))
        qkv_bias = torch.cat(
            (self.q_bias, self.qkv.weight.new_zeros(d_model), value_bias)
        )
>>>>>>> REPLACE