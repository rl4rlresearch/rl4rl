MECHANISM: Three-entry query/key basis gauge fixing

HYPOTHESIS: Fixing a third query-weight coordinate at zero on the qualified 1,344-parameter design will retain at least 99% accuracy after 45,000 updates while reducing the model to 1,343 parameters.

INTENDED_EDIT: Extend the existing query-weight constraint from the first two flattened entries to the first three, reconstructing the full QKV matrix with three leading zeros while preserving initialization RNG consumption.

EVIDENCE: The current design achieved 100% accuracy with two fixed query-weight coordinates at 1,344 parameters, after the one-coordinate design also achieved 100%; this makes the next incremental coordinate of the same Q/K basis gauge the most direct informative reduction.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit two
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 2))
=======
        # Key and value biases are redundant. Q/K basis changes permit three
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (2, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (3, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    module.weight.copy_(full_weight[2:])
=======
                    module.weight.copy_(full_weight[3:])
>>>>>>> REPLACE