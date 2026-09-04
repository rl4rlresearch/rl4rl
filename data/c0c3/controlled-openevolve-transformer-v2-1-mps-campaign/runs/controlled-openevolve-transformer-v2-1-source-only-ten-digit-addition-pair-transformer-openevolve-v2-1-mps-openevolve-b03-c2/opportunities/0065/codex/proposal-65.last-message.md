MECHANISM: Additional first-head query-bias basis gauge

HYPOTHESIS: Fixing a second first-head query-bias coordinate will reduce the qualified model to 1496 parameters while retaining at least 99% accuracy, because the first head has unused query/key basis freedom and does not use the failed additional query-row scale chart.

INTENDED_EDIT: Omit the first head’s penultimate query-bias coordinate and reconstruct the final two first-head coordinates and final second-head coordinate as zero.

EVIDENCE: The current per-head query-bias gauge achieved 99.93% at 1497 parameters, while the 1496 dual-scale chart achieved only 82.73%; this tests an unused basis-direction gauge without imposing another normalized query-weight chart.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Independent query/key basis gauges fix each head's
        # final query-bias coordinate; the second zero-bias query coordinate
        # additionally uses a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final two query-bias
        # coordinates in the first head and the final coordinate in the
        # second; the latter also uses a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias_split = self.head_dim - 1
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(1),
            )
        )
=======
        bias_split = self.head_dim - 2
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(2),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(1),
            )
        )
>>>>>>> REPLACE