MECHANISM: Seven-entry query/key basis gauge fixing

HYPOTHESIS: Fixing a seventh leading query-weight coordinate at zero will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,340 to 1,339 learned parameters.

INTENDED_EDIT: Extend the existing query-weight constraint from six to seven leading flattened entries, reconstructing the full QKV matrix with seven fixed zeros while preserving full-matrix initialization RNG consumption.

EVIDENCE: The six-entry design achieved 100% accuracy with 1,340 parameters, following 100% results with one through five fixed entries; this directly motivates testing the next scalar in the same Q/K basis gauge.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit six
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 6))
=======
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (6, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    module.weight.copy_(full_weight[6:])
=======
                    module.weight.copy_(full_weight[7:])
>>>>>>> REPLACE