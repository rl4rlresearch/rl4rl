MECHANISM: Incremental initialization-preserving positional-row shift gauge

HYPOTHESIS: Anchoring a second positional-embedding row’s uniform-shift coordinate will reduce the model from 1,598 to 1,597 parameters while retaining at least 99% accuracy after 19,000 steps, because each positional row has the same exact shift symmetry and the first anchored row achieved 99.94%.

INTENDED_EDIT: Store the penultimate positional row as seven relative coordinates plus a zero anchor, preserve its full initialization and implicit eight-coordinate AdamW updates, and add 1,000 low-rate refinement steps.

EVIDENCE: Initialization-preserving anchoring of the final positional row achieved 99.94% at 1,598 parameters after 18,000 steps, directly validating this symmetry and optimizer treatment for one positional row.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_last = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_penultimate = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_last = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[:-1].detach().clone()
            )
            full_pos_row.sub_(full_pos_row[-1].clone())
            self.pos_emb_last.copy_(full_pos_row[:-1])
=======
            full_pos_penultimate = self.pos_emb.weight[-2].detach().clone()
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[:-2].detach().clone()
            )
            full_pos_penultimate.sub_(full_pos_penultimate[-1].clone())
            self.pos_emb_penultimate.copy_(full_pos_penultimate[:-1])
            full_pos_row.sub_(full_pos_row[-1].clone())
            self.pos_emb_last.copy_(full_pos_row[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _position_weight(self) -> torch.Tensor:
        last_row = torch.cat(
            (self.pos_emb_last, self.pos_emb_last.new_zeros(1))
        ).unsqueeze(0)
        return torch.cat((self.pos_emb.weight, last_row), dim=0)
=======
    def _position_weight(self) -> torch.Tensor:
        penultimate_row = torch.cat(
            (
                self.pos_emb_penultimate,
                self.pos_emb_penultimate.new_zeros(1),
            )
        ).unsqueeze(0)
        last_row = torch.cat(
            (self.pos_emb_last, self.pos_emb_last.new_zeros(1))
        ).unsqueeze(0)
        return torch.cat(
            (self.pos_emb.weight, penultimate_row, last_row), dim=0
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [model.token_emb.weight, model.pos_emb_last] + [
=======
    gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb_penultimate,
        model.pos_emb_last,
    ] + [
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=18000)
=======
    p.add_argument("--train-steps", type=int, default=19000)
>>>>>>> REPLACE