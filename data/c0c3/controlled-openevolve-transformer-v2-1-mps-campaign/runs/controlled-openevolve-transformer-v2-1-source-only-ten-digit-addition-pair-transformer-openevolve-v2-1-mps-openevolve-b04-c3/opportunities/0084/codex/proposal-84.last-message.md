MECHANISM: Tied-token residual-basis gauge fixing

HYPOTHESIS: The qualified 1,146-parameter multi-query model will retain at least 99% accuracy after 45,000 updates when one coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,145 learned parameters.

INTENDED_EDIT: Adopt the qualified shared-key/value attention, eight-scalar Fourier constraint, and two-coordinate attention-output-bias constraint, then learn all but one scalar of the tied token embedding while preserving full initialization RNG consumption.

EVIDENCE: The 1,146-parameter multi-query design achieved 100% accuracy, while further reducing positional mixing or attention-output bias failed; this tests a different redundancy, fixing one coordinate under the remaining orthogonal basis freedom of the tied token subspace.

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
        # The two token-free residual channels provide one rotational gauge.
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
        # The final two attention-output bias coordinates are fixed at zero.
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
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        # Choose a basis for the tied token subspace by fixing its final scalar.
        # The full Embedding constructor above preserves constructor RNG use.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 1)
        )
        self.token_emb.fixed_weight_trim = (0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final six scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 6)
        )
        self.pos_mix.fixed_weight_trim = (0, 6)
=======
        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final eight scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
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

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, nn.Embedding):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.num_embeddings * module.embedding_dim
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
=======
        token_weight = F.pad(self.token_emb.weight, (0, 1)).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
        token_x = F.pad(
            F.embedding(idx, token_weight),
            (0, self.cfg.d_model - self.token_dim),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 6)).view(
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.token_scale * self.token_emb.weight,
=======
            self.token_scale * token_weight,
>>>>>>> REPLACE