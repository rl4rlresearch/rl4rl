MECHANISM: Pre-attention LayerNorm scale gauge fixing

HYPOTHESIS: Fixing the pre-attention LayerNorm scale to one will reduce the qualified model from 1,340 to 1,332 parameters while retaining at least 99% accuracy, because its learned scale can be absorbed into the unrestricted QKV input columns without reducing the attention function class.

INTENDED_EDIT: Disable the pre-attention LayerNorm’s eight learned scale parameters while preserving all five external bias coefficients.

EVIDENCE: The current 1,340-parameter model achieved 100% accuracy, and prior successful pre-attention gauge reductions established that query bias, key-softmax invariance, and attention-output bias absorb constant effects; fixing the scale also preserves its initial value and leaves QKV free to learn the equivalent column scaling.

<<<<<<< SEARCH
        self.norm = nn.LayerNorm(normalized_shape, bias=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))
=======
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.bias_coeff = nn.Parameter(torch.zeros(normalized_shape - 3))
>>>>>>> REPLACE