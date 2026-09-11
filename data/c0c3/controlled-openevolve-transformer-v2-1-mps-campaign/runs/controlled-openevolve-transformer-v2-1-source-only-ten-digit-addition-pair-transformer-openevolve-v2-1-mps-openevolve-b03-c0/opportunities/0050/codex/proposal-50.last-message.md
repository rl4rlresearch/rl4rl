MECHANISM: Initialization-preserving global token-embedding gauge

HYPOTHESIS: Anchoring one scalar of the tied token-embedding/output matrix, while sampling an equivalent full initialization and preserving implicit full-coordinate AdamW updates, will reduce the model from 1,602 to 1,601 parameters and retain at least 99% accuracy after 15,000 steps.

INTENDED_EDIT: Store 103 of the 104 tied embedding coordinates, reconstruct the final coordinate as zero for both input embedding and output logits, initialize the anchored representation from a full i.i.d. matrix, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: Gauge-aware anchoring of the attention projection bias achieved 99.89% at 1,607 parameters, whereas positional-embedding anchoring reached only 56.28%; preserving both the full initialization distribution and implicit optimizer state tests whether initialization geometry caused that failure. A global scalar shift of the tied token matrix is an exact symmetry because it propagates as a uniform residual shift removed by LayerNorm and adds only a common offset to all logits.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * cfg.d_model - 1)
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            full_token_weight = self.token_emb.weight.new_empty(
                cfg.vocab_size * cfg.d_model
            )
            nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
            full_token_weight.sub_(full_token_weight[-1].clone())
            self.token_emb.weight.copy_(full_token_weight[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
=======
    def _token_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.token_emb.weight, self.token_emb.weight.new_zeros(1))
        ).view(self.cfg.vocab_size, self.cfg.d_model)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        token_weight = self._token_weight()
        x = F.embedding(idx, token_weight) + self.pos_emb(pos)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [block.attn.proj.bias for block in model.blocks]
=======
    gauge_parameters = [model.token_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=14000)
=======
    p.add_argument("--train-steps", type=int, default=15000)
>>>>>>> REPLACE