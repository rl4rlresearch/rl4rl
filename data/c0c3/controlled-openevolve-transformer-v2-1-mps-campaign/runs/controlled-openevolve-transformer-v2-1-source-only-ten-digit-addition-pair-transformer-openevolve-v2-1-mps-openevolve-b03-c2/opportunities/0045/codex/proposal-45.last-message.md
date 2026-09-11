MECHANISM: Per-head query-key basis gauge fixing

HYPOTHESIS: Fixing one query-bias coordinate in each attention head will reduce the verified 1505-parameter model to 1504 parameters while retaining at least 99% accuracy, because each head has an independent invertible query/key basis symmetry.

INTENDED_EDIT: Store three query-bias coordinates per four-dimensional head and reconstruct the fourth coordinate of each head as zero.

EVIDENCE: Fixing one query-bias coordinate already achieved 100% accuracy at 1505 parameters; applying the same gauge independently to the other head is the smallest symmetry-grounded extension.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Within each head, an invertible query/key basis change
        # can additionally fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbable into the
        # output bias. Each head has an independent invertible query/key basis
        # change that can fix one generic query-bias coordinate to zero.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - n_head))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(1))
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias.view(
                    self.n_head, self.head_dim - 1
                ),
                self.qkv.bias.new_zeros((self.n_head, 1)),
            ),
            dim=-1,
        ).reshape(-1)
>>>>>>> REPLACE