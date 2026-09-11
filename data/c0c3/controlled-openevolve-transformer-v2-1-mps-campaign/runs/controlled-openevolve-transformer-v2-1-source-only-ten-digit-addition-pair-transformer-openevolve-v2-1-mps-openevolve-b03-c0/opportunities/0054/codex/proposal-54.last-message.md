MECHANISM: Initialization-preserving positional-row shift gauge

HYPOTHESIS: Anchoring one positional-embedding row’s uniform-shift coordinate while preserving its full initialization and implicit eight-coordinate AdamW updates will reduce the model from 1,599 to 1,598 parameters and retain at least 99% accuracy after 18,000 steps.

INTENDED_EDIT: Store the final positional row as seven relative coordinates plus a zero anchor, reconstruct it during forward passes, optimize it with GaugeAdamW, and add 1,000 low-rate refinement steps.

EVIDENCE: A naive positional-embedding anchor reached only 56.28%, but initialization-preserving gauge optimization rescued the attention-projection column mechanism from 40.05% to 99.98%; the same treatment directly tests whether the earlier positional failure was caused by initialization and optimizer geometry.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_last = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
            full_token_weight.sub_(full_token_weight[-1].clone())
            self.token_emb.weight.copy_(full_token_weight[:-1])
=======
            nn.init.normal_(full_token_weight, mean=0.0, std=0.02)
            full_token_weight.sub_(full_token_weight[-1].clone())
            self.token_emb.weight.copy_(full_token_weight[:-1])

            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[:-1].detach().clone()
            )
            full_pos_row.sub_(full_pos_row[-1].clone())
            self.pos_emb_last.copy_(full_pos_row[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _token_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.token_emb.weight, self.token_emb.weight.new_zeros(1))
        ).view(self.cfg.vocab_size, self.cfg.d_model)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
=======
    def _token_weight(self) -> torch.Tensor:
        return torch.cat(
            (self.token_emb.weight, self.token_emb.weight.new_zeros(1))
        ).view(self.cfg.vocab_size, self.cfg.d_model)

    def _position_weight(self) -> torch.Tensor:
        last_row = torch.cat(
            (self.pos_emb_last, self.pos_emb_last.new_zeros(1))
        ).unsqueeze(0)
        return torch.cat((self.pos_emb.weight, last_row), dim=0)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = self._token_weight()
        x = F.embedding(idx, token_weight) + self.pos_emb(pos)
=======
        token_weight = self._token_weight()
        position_weight = self._position_weight()
        x = F.embedding(idx, token_weight) + F.embedding(pos, position_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [model.token_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
=======
    gauge_parameters = [model.token_emb.weight, model.pos_emb_last] + [
        block.attn.proj.bias for block in model.blocks
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=17000)
=======
    p.add_argument("--train-steps", type=int, default=18000)
>>>>>>> REPLACE