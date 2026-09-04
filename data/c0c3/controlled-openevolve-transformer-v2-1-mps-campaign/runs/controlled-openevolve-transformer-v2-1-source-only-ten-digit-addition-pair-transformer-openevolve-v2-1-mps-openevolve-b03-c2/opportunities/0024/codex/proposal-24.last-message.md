MECHANISM: One-coordinate pre-LayerNorm scale gauge

HYPOTHESIS: Extending the verified 1595-parameter design by fixing one `ln2` scale coordinate at unity will produce a 1594-parameter model with at least 99% accuracy, because the omitted common scale can be absorbed by `fc1` while preserving the initialized forward function.

INTENDED_EDIT: Adopt the qualified tied-token quotient and biasless block LayerNorms, then store seven learned `ln2` scale coordinates, reconstruct the eighth as one, and retain quotient-aware optimization for token, positional, and MLP-output parameters.

EVIDENCE: The 1595-parameter design achieved 99.42% accuracy after token quotienting and removal of both block LayerNorm biases; fixing one `ln2` scale coordinate is the smallest next reduction and exploits its exact multiplicative redundancy with the following learned `fc1` matrix.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = None
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln2.bias = None
        # A common scale of ln2.weight can be absorbed into fc1, so fix its
        # final coordinate at one and learn the remaining relative scales.
        self.ln2.weight = nn.Parameter(torch.ones(cfg.d_model - 1))
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(1))
        )
        ln2_x = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            ln2_weight,
            None,
            self.ln2.eps,
        )
        x = x + self.mlp(ln2_x)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Uniform feature offsets in positional rows are canceled by all
        # downstream LayerNorms. Store only relative coordinates per row.
=======
        self.apply(self._init_weights)

        # A global feature-uniform shift of the tied token table is canceled
        # at the input and produces only a uniform output-logit shift.
        full_token_weight = self.token_emb.weight.detach().flatten()
        self.token_emb.weight = nn.Parameter(
            full_token_weight[:-1] - full_token_weight[-1]
        )
        self.lm_head.weight = self.token_emb.weight

        # Uniform feature offsets in positional rows are canceled by all
        # downstream LayerNorms. Store only relative coordinates per row.
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)

        token_relative = torch.cat(
            (
                self.token_emb.weight,
                self.token_emb.weight.new_zeros(1),
            )
        )
        token_weight = (
            token_relative + self.token_emb.weight.mean()
        ).view(self.cfg.vocab_size, self.cfg.d_model)

        pos_relative = torch.cat(
            (
                self.pos_emb.weight,
                self.pos_emb.weight.new_zeros(
                    (self.pos_emb.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        pos_weight = pos_relative + self.pos_emb.weight.mean(
            dim=-1, keepdim=True
        )
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
>>>>>>> REPLACE