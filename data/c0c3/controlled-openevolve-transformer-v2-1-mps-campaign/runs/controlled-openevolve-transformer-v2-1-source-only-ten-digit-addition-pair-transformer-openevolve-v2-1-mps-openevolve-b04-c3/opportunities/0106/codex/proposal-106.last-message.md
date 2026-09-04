MECHANISM: Near-constant Fourier coefficient pruning

HYPOTHESIS: Fixing the ninth trailing positional-synthesis coefficient at zero will reduce the qualified 1,138-parameter model to 1,137 parameters while retaining at least 99% accuracy after 45,000 updates.

INTENDED_EDIT: Adopt the qualified factorized shared-key/value attention backbone and its established constraints, then prune one additional positional coefficient corresponding to the highest-index cosine feature while preserving full constructor RNG consumption.

EVIDENCE: The qualified 1,138-parameter design achieved 99.78% accuracy with eight positional coefficients fixed; extending that constraint by one scalar probes an untested subsystem after additional attention-output constraints failed sharply.

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
        # share one learned key/value dictionary. Seven query coordinates and
        # one trailing value coefficient are fixed. Constructing the original
        # Linear preserves constructor RNG consumption.
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
        # Fix three trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 3))
        self.proj.fixed_weight_trim = (0, 3)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
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
        key_weight = key_weight.view(self.head_dim, d_model)
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model)
        q = F.linear(x, query_weight, query_bias)
        shared_kv = F.linear(
            x,
            torch.cat((key_weight, value_weight), dim=0),
        )
        k, v = shared_kv.chunk(2, dim=-1)
        k = k.repeat(1, 1, self.n_head)
        v = v.repeat(1, 1, self.n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 1))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_weight = F.pad(self.proj.weight, (0, 3)).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final seven scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 7)
        )
        self.pos_mix.fixed_weight_trim = (0, 7)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        # Choose a basis for the tied token subspace by fixing its final four
        # scalars. Constructing the full Embedding preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)

        # The qualified final zero output row is extended by fixing one
        # additional coefficient of the dense Fourier synthesizer.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 9)
        )
        self.pos_mix.fixed_weight_trim = (0, 9)
>>>>>>> REPLACE

<<<<<<< SEARCH
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
                if hasattr(module, "shared_kv_dim"):
                    d_model = module.in_features
                    query_weight, key_weight, value_weight = full_weight.view(
                        3, d_model, d_model
                    )
                    compact_weight = torch.cat(
                        (
                            query_weight.flatten()[module.query_weight_trim :],
                            key_weight[: module.shared_kv_dim].flatten(),
                            value_weight[: module.shared_kv_dim]
                            .flatten()[: -module.shared_value_weight_trim],
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
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
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
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
        token_x = F.pad(
            F.embedding(idx, token_weight),
            (0, self.cfg.d_model - self.token_dim),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = F.pad(self.pos_mix.weight, (0, 7)).view(
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 9)).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.token_scale * self.token_emb.weight,
=======
            self.token_scale * token_weight,
>>>>>>> REPLACE