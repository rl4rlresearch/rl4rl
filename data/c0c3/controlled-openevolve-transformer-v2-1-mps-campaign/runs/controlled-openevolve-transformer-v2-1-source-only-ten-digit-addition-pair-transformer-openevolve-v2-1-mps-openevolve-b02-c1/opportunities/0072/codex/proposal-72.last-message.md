MECHANISM: Rank-one content routing with rank-four value transport

HYPOTHESIS: Reducing both heads’ query/key score rank from two to one will lower the model from 1,002 to 978 parameters while retaining at least 99% accuracy, because rank-two routing achieved 99.93% and independently learned relative-position biases remain the load-bearing routing pathway.

INTENDED_EDIT: Use one learned query/key factor per attention head while preserving four-dimensional values, relative-bias tables, gauge-aware optimization, initialization behavior, and generic autoregressive decoding.

EVIDENCE: Rank-three routing achieved 99.85% and rank-two routing improved to 99.93%, whereas replacing the independent relative-position tables with affine positional pointers produced 0% accuracy; this supports directly testing whether content routing can be compressed another rank without disturbing positional routing.

<<<<<<< SEARCH
        self.score_dim = self.head_dim - 2
=======
        self.score_dim = self.head_dim - 3
>>>>>>> REPLACE