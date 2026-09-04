MECHANISM: RNG-aligned shared-key multi-head attention

HYPOTHESIS: The model does not require a separate key subspace for each attention head: sharing one learned 4-dimensional key projection while retaining head-specific queries and values will reduce parameters from 1,612 to 1,580 and maintain at least 99% accuracy.

INTENDED_EDIT: Replace the 24-row QKV weight with compact query, shared-key, and value projections; broadcast the shared keys across heads; and preserve the original initialization RNG stream.

EVIDENCE: Narrowing the feed-forward network caused severe accuracy loss, while the current full-width model reached 99.96%. The key-bias experiments also showed initialization alignment matters, motivating a different attention mechanism that preserves MLP capacity, query/value diversity, and downstream initialization.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query and value portions of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
=======
        # Construct the original affine first to preserve the constructor RNG
        # stream, then compact its learned weight to Q + shared K + V.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.weight = nn.Parameter(
            self.qkv.weight.new_empty(2 * d_model + self.head_dim, d_model)
        )
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
        self.qkv.out_features = 2 * d_model + self.head_dim
        self.qkv._shared_key_d_model = d_model
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = torch.split(
            qkv, (d_model, self.head_dim, d_model), dim=-1
        )
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, 1, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            shared_key_d_model = getattr(module, "_shared_key_d_model", None)
            if shared_key_d_model is None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                full_weight = module.weight.new_empty(
                    3 * shared_key_d_model, shared_key_d_model
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                head_dim = module.weight.size(0) - 2 * shared_key_d_model
                compact_weight = torch.cat(
                    (
                        full_weight[:shared_key_d_model],
                        full_weight[
                            shared_key_d_model : shared_key_d_model + head_dim
                        ],
                        full_weight[2 * shared_key_d_model :],
                    ),
                    dim=0,
                )
                with torch.no_grad():
                    module.weight.copy_(compact_weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE