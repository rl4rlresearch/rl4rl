MECHANISM: Complete tied-embedding rotational gauge fixing

HYPOTHESIS: Extending the verified 1,139-parameter design by fixing a fifth trailing token-embedding scalar will retain at least 99% accuracy while reducing the model to 1,138 learned parameters.

INTENDED_EDIT: Adopt Reference Design 2’s qualified shared key/value attention and projection constraints, then align the final tied token vector to one learned coordinate by increasing its trim from four scalars to five.

EVIDENCE: Reference Design 2 reached 1.0 accuracy with 1,139 parameters and four fixed token-embedding scalars; Reference Design 3 also reached 1.0 with three. The failed 1,138-parameter attempt instead extended the attention-output trim and fell to 0.9382, motivating reduction along the independently qualified embedding-basis gauge.

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
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
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
        # Fix two trailing coefficients on the output side of the shared-value
        # basis while preserving the full Linear's constructor RNG consumption.
        self.proj.weight = nn.Parameter(torch.empty(d_model * d_model - 2))
        self.proj.fixed_weight_trim = (0, 2)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
        qkv = F.linear(x, qkv_weight, qkv_bias)
=======
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
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_weight = F.pad(
            value_weight,
            (0, self.qkv.shared_value_weight_trim),
        ).view(self.head_dim, d_model).repeat(self.n_head, 1)
        qkv_weight = torch.cat((query_weight, key_weight, value_weight), dim=0)
        qkv = F.linear(x, qkv_weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 1))
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_weight = F.pad(self.proj.weight, (0, 2)).view(d_model, d_model)
        proj_bias = F.pad(self.proj.bias, (0, 2))
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Token identities use a learned rank-six code. A fixed zero lift places
        # that code in the full residual stream without adding parameters; the
        # same lifted matrix is used by the output classifier below.
        self.token_dim = cfg.d_model - 2
        if self.token_dim <= 0:
            raise ValueError("d_model must be at least 3")
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
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
        # Learn token identities in a six-dimensional subspace, then lift them
        # into the eight-dimensional residual stream without extra parameters.
        # The same learned code is used by the output classifier.
        self.token_dim = cfg.d_model - 2
        if self.token_dim <= 0:
            raise ValueError("d_model must be at least 3")
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        # Complete the rotational basis choice for the final tied token vector
        # by fixing five of its six coordinates. Constructing the full
        # Embedding above preserves constructor RNG consumption.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 5)
        )
        self.token_emb.fixed_weight_trim = (0, 5)

        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 8)
        )
        self.pos_mix.fixed_weight_trim = (0, 8)
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
        token_weight = F.pad(self.token_emb.weight, (0, 5)).view(
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
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.token_scale * self.token_emb.weight,
=======
            self.token_scale * token_weight,
>>>>>>> REPLACE