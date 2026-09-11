MECHANISM: Balanced per-head query-bias basis gauge

HYPOTHESIS: Fixing a second query-bias coordinate in the second head will reduce the qualified 1496-parameter design to 1495 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified two-coordinate first-head bias gauge, then omit the penultimate second-head query bias so each head reconstructs its final two query-bias coordinates as zero.

EVIDENCE: Fixing a second first-head query-bias coordinate achieved 99.98% accuracy at 1496 parameters, whereas the alternative 1496 dual query-row scale chart reached only 82.73%; applying the successful basis gauge symmetrically to the second head is the closest supported reduction.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The final zero-bias query coordinate additionally uses
        # a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in each head; the final coordinate additionally uses
        # a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
=======
        bias_split = self.head_dim - 2
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE