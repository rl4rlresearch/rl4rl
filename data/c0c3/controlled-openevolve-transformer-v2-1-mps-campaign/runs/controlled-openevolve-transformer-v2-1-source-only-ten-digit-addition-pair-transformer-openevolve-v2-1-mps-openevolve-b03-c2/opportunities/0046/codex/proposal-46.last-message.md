MECHANISM: Single-coordinate query-key basis gauge fixing

HYPOTHESIS: Fixing one query-bias coordinate to zero will reduce the current model from 1506 to 1505 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Store seven learned query-bias coordinates and reconstruct the eighth as zero, leaving the protected decoding interface and training procedure unchanged.

EVIDENCE: Reference Design 1 verified this exact 1505-parameter gauge at 100% accuracy; the 1504-parameter per-head extension reached only 98.61%, identifying the single-coordinate version as the qualified boundary.

<<<<<<< SEARCH
        # Key bias is softmax-invariant, and every value-bias coordinate can
        # be absorbed by the downstream projection bias. Store only query
        # bias and reconstruct the other two bias vectors in fixed gauges.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Within each head, an invertible query/key basis change
        # can additionally fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = self.qkv.bias
=======
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
>>>>>>> REPLACE