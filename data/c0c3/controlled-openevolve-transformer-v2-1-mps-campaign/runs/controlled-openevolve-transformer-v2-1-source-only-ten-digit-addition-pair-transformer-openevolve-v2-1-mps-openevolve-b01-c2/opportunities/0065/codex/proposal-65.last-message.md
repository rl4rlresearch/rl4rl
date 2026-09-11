MECHANISM: Balanced per-head Q/K basis query-bias gauge

HYPOTHESIS: Fixing local query-bias coordinate 2 in both attention heads will produce a 1,575-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain the successful first-head query-bias coordinate-2 constraint and additionally fix global query-bias coordinate 6, the corresponding coordinate in the second head.

EVIDENCE: Fixing first-head coordinate 2 achieved 99.1% accuracy at 1,576 parameters, while balanced query-row gauges across both heads previously achieved 99.93%; applying the same successful basis coordinate symmetrically is the smallest informative reduction.

<<<<<<< SEARCH
        # Query coordinate 2 is fixed; key and value biases remain absent.
        self.bias = nn.Parameter(base.bias.new_empty(d_model - 1))
=======
        # Local query coordinate 2 is fixed in both attention heads; key and
        # value biases remain absent.
        self.fixed_query_biases = (2, head_dim + 2)
        self.bias = nn.Parameter(
            base.bias.new_empty(d_model - len(self.fixed_query_biases))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.bias[:2],
                self.bias.new_zeros(1),
                self.bias[2:],
            )
        )
=======
        query_bias_parts = []
        bias_index = 0
        for coordinate in range(self.in_features):
            if coordinate in self.fixed_query_biases:
                query_bias_parts.append(self.bias.new_zeros(()))
            else:
                query_bias_parts.append(self.bias[bias_index])
                bias_index += 1
        query_bias = torch.stack(query_bias_parts)
>>>>>>> REPLACE