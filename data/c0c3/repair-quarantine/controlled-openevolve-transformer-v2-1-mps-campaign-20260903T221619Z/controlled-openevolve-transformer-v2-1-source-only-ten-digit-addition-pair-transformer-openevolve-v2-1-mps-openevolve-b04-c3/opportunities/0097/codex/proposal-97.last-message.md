MECHANISM: Independent query-coordinate scale gauge normalization

HYPOTHESIS: Extending the verified 1,138-parameter design by fixing a second query-coordinate pivot at 0.02 will retain at least 99% accuracy with 1,137 learned parameters, because independently rescaling that query coordinate and the reciprocal shared-key coordinate preserves attention scores.

INTENDED_EDIT: Adopt the qualified two-coefficient attention-output constraint and reconstruct the first two query pivots as fixed nonzero values, while preserving full constructor initialization RNG consumption.

EVIDENCE: The 1,138-parameter reference achieved 100% accuracy after fixing the first query pivot through query/key scale freedom; applying the same independent gauge to the next query coordinate is the smallest evidence-backed reduction.

<<<<<<< SEARCH
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are gauge-fixed. Constructing the
        # original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.shared_value_weight_trim = 1
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Fix one coefficient on the output side of the shared-value basis.
        # Constructing the full Linear preserves constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 1))
        self.proj.fixed_weight_trim = (0, 1)
=======
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coefficients are
        # zero, the next two coordinate pivots are fixed nonzero, and one
        # trailing value coefficient is zero. Constructing the original Linear
        # preserves constructor RNG.
        query_weight_size = d_model * d_model - 9
        shared_key_weight_size = self.head_dim * d_model
        shared_value_weight_size = self.head_dim * d_model - 1
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_weight_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.query_fixed_pivots = 2
        self.qkv.query_pivot = 0.02
        self.qkv.shared_value_weight_trim = 1
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Fix two trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 2))
        self.proj.fixed_weight_trim = (0, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
=======
        query_weight_size = (
            d_model * d_model
            - self.qkv.query_weight_trim
            - self.qkv.query_fixed_pivots
        )
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (
                query_weight_size,
                kv_weight_size,
                kv_weight_size - self.qkv.shared_value_weight_trim,
            ),
        )
        query_weight = torch.cat(
            (
                query_weight.new_zeros(self.qkv.query_weight_trim),
                query_weight.new_full(
                    (self.qkv.query_fixed_pivots,),
                    self.qkv.query_pivot,
                ),
                query_weight,
            )
        ).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_weight = F.pad(self.proj.weight, (0, 1)).view(d_model, d_model)
=======
        proj_weight = F.pad(self.proj.weight, (0, 2)).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                            query_weight.flatten()[module.query_weight_trim :],
=======
                            query_weight.flatten()[
                                module.query_weight_trim
                                + module.query_fixed_pivots :
                            ],
>>>>>>> REPLACE