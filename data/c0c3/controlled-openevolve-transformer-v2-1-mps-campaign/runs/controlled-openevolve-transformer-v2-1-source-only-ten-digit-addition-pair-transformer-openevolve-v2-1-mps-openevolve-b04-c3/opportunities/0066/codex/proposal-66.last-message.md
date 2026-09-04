MECHANISM: Four-entry query/key basis gauge fixing

HYPOTHESIS: Fixing a fourth query-weight coordinate at zero on the qualified 1,343-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,342 learned parameters.

INTENDED_EDIT: Extend the query-weight constraint from three to four leading flattened entries, reconstructing the full QKV matrix with four fixed zeros while preserving initialization RNG consumption.

EVIDENCE: The three-entry design achieved 100% accuracy with 1,343 parameters, following 100% results with one and two fixed entries; this supports testing the next incremental coordinate of the same Q/K basis gauge.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit one
        # query-weight entry to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))
=======
        # Key and value biases are redundant. Q/K basis changes permit four
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (1, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (4, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    module.weight.copy_(full_weight[1:])
=======
                    module.weight.copy_(full_weight[4:])
>>>>>>> REPLACE