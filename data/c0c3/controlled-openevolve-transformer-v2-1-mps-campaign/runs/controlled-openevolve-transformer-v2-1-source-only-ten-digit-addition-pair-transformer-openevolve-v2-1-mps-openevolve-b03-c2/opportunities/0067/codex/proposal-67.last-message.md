MECHANISM: Asymmetric third query-bias basis gauge

HYPOTHESIS: Fixing a third first-head query-bias coordinate will reduce the qualified 1495-parameter design to 1494 parameters while retaining at least 99% accuracy, because the first head previously tolerated the additional bias-basis restriction at 99.98% accuracy and retains one free affine query-bias coordinate.

INTENDED_EDIT: Reproduce the balanced two-coordinate-per-head bias gauge and additionally omit the first head’s second query-bias coordinate, reconstructing three trailing first-head coordinates and two trailing second-head coordinates as zero.

EVIDENCE: The balanced query-bias design reached 99.93% accuracy at 1495 parameters, and introducing the second fixed coordinate in the first head reached 99.98%; this makes extending the same first-head basis gauge the closest supported one-parameter reduction.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. The final zero-bias query coordinate additionally uses
        # a diagonal query/key scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Query/key basis gauges fix the final three query-bias
        # coordinates in the first head and the final two in the second; the
        # latter's final coordinate also uses a diagonal scale gauge.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
=======
        bias_split = self.head_dim - 3
        query_bias = torch.cat(
            (
                self.qkv.bias[:bias_split],
                self.qkv.bias.new_zeros(3),
                self.qkv.bias[bias_split:],
                self.qkv.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE