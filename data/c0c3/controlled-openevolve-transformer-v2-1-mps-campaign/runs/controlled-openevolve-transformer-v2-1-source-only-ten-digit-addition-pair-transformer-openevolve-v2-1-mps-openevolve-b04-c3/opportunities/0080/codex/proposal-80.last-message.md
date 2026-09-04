MECHANISM: Two-query multi-query attention with a shared learned key/value dictionary

HYPOTHESIS: Sharing one learned key and value projection across both attention heads, while retaining distinct query projections, will achieve at least 99% accuracy after 45,000 updates with 1,147 parameters.

INTENDED_EDIT: Challenge the assumption that each routing head needs independent key/value features. Adopt the qualified eight-entry Fourier constraint and replace independent per-head K/V projections with one shared learned K/V head, preserving distinct queries and full-matrix initialization RNG consumption.

EVIDENCE: The 1,211-parameter eight-entry Fourier design achieved 100% accuracy, while restricting positional mixing and reducing token-code dimension failed at 46.49% and 56.64%. This preserves those load-bearing representations and instead compresses attention: both routing heads process the same token-position stream, so distinct queries can select different sources while sharing their learned key/value representation.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources, while both heads
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
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final two scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 2)
        )
        self.pos_mix.fixed_weight_trim = (0, 2)
=======
        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 8)
        )
        self.pos_mix.fixed_weight_trim = (0, 8)
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 2)).view(
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
>>>>>>> REPLACE