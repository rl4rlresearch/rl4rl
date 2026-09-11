MECHANISM: Canonical-basis shared-value attention

HYPOTHESIS: Fixing the token-facing 4×4 block of the shared value projection to a scaled identity will preserve at least 99% accuracy after 45,000 updates with 1,126 parameters, because the dense output projection can absorb an invertible change of the four-dimensional value basis.

INTENDED_EDIT: Replace the assumption that all 32 shared-value coefficients must be learned with a canonical 0.02-scaled passthrough basis plus 16 learned coefficients for the remaining input channels; also adopt the qualified four-scalar tied-token constraint.

EVIDENCE: The shared-key/value model achieved 100% accuracy, and the four-scalar token constraint separately achieved 100% at 1,142 parameters. Unlike failed reductions of positional capacity, token rank, or output biases, this removes a 16-parameter value/output basis redundancy while retaining the full residual width, positional map, MLP, attention routing, and dense learned output projection.

<<<<<<< SEARCH
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
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Distinct learned queries select different sources while both heads
        # share one key/value dictionary. The value dictionary uses a canonical
        # token-facing basis: its square leading block is fixed to 0.02 I, and
        # only the remaining channel couplings are learned. The dense output
        # projection can absorb this otherwise redundant value-basis choice.
        # Constructing the original Linear preserves constructor RNG consumption.
        query_weight_size = d_model * d_model - 7
        shared_key_weight_size = self.head_dim * d_model
        shared_value_tail_size = self.head_dim * (d_model - self.head_dim)
        self.qkv.weight = nn.Parameter(
            torch.empty(
                query_weight_size
                + shared_key_weight_size
                + shared_value_tail_size
            )
        )
        self.qkv.shared_kv_dim = self.head_dim
        self.qkv.query_weight_trim = 7
        self.qkv.value_anchor_dim = self.head_dim
        self.qkv.value_anchor_scale = 0.02
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        query_weight_size = d_model * d_model - self.qkv.query_weight_trim
        key_weight_size = self.head_dim * d_model
        value_tail_size = self.head_dim * (
            d_model - self.qkv.value_anchor_dim
        )
        query_weight, key_weight, value_tail = torch.split(
            self.qkv.weight,
            (query_weight_size, key_weight_size, value_tail_size),
        )
        query_weight = F.pad(
            query_weight,
            (self.qkv.query_weight_trim, 0),
        ).view(d_model, d_model)
        key_weight = key_weight.view(self.head_dim, d_model).repeat(
            self.n_head, 1
        )
        value_anchor = torch.eye(
            self.qkv.value_anchor_dim,
            device=value_tail.device,
            dtype=value_tail.dtype,
        ) * self.qkv.value_anchor_scale
        value_weight = torch.cat(
            (
                value_anchor,
                value_tail.view(
                    self.head_dim,
                    d_model - self.qkv.value_anchor_dim,
                ),
            ),
            dim=1,
        ).repeat(self.n_head, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                            value_weight[
                                : module.shared_kv_dim,
                                module.value_anchor_dim :,
                            ].flatten(),
                        )
                    )
                    with torch.no_grad():
                        module.weight.copy_(compact_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Choose a basis for the tied token subspace by fixing its final three
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 3)
        )
        self.token_emb.fixed_weight_trim = (0, 3)
=======
        # Use the qualified tied-token basis with its final four scalars fixed.
        # Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 3)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
>>>>>>> REPLACE