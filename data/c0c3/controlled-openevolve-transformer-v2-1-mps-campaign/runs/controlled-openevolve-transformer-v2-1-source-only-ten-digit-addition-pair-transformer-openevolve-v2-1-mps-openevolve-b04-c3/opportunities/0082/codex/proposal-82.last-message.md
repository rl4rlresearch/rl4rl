MECHANISM: Shared-key/value multi-query attention with a two-coordinate output-bias constraint

HYPOTHESIS: The verified 1,147-parameter multi-query model will retain at least 99% accuracy when a second attention-output bias scalar is fixed at zero, yielding 1,146 learned parameters after 45,000 updates.

INTENDED_EDIT: Adopt the qualified shared key/value attention design with distinct queries, retain the successful eight-scalar Fourier constraint, and learn six rather than seven attention-output bias coordinates while preserving constructor RNG consumption.

EVIDENCE: Shared key/value attention achieved 100% accuracy with 1,147 parameters. The ninth positional-mixer constraint failed at 33.92%, so the next one-parameter probe targets a different redundancy; every qualified backbone already fixes one attention-output bias coordinate without loss.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Token codes occupy only the first d_model - 2 channels. An orthogonal
        # rotation of the remaining two residual channels can align this bias
        # with one axis, allowing its final coordinate to be fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources while both heads
        # share one learned key/value dictionary. Seven query coordinates remain
        # gauge-fixed. Constructing the original Linear preserves constructor RNG.
        query_weight_size = d_model * d_model - 7
        shared_kv_weight_size = 2 * self.head_dim * d_model
        self.qkv.weight = nn.Parameter(
            torch.empty(query_weight_size + shared_kv_weight_size)
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
        # Retain only six learned coordinates of the additive attention-output
        # bias; its final two coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
=======
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        kv_weight_size = self.head_dim * d_model
        query_weight, key_weight, value_weight = torch.split(
            self.qkv.weight,
            (query_weight_size, kv_weight_size, kv_weight_size),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = value_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 1))
=======
        proj_bias = F.pad(self.proj.bias, (0, 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
            else:
=======
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                if hasattr(module, "shared_kv_dim"):
                    d_model = module.in_features
                    query_weight, key_weight, value_weight = full_weight.view(
                        3, d_model, d_model
                    )
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim].flatten(),
                        )
                    )
                    with torch.no_grad():
                        module.weight.copy_(compact_weight)
                else:
                    start, end_trim = module.fixed_weight_trim
                    end = full_weight.numel() - end_trim
                    with torch.no_grad():
                        module.weight.copy_(full_weight[start:end])
            else:
>>>>>>> REPLACE