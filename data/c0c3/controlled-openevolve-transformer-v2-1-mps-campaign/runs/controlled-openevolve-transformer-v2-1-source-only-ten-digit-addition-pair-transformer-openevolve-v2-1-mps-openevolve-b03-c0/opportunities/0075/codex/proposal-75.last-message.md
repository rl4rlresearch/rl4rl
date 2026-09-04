MECHANISM: Nonadjacent trajectory-preserving positional-row shift gauge

HYPOTHESIS: Gauge-anchoring initialized positional row 0 will reduce the model from 1,583 to 1,582 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split positional row 0 into seven learned relative coordinates plus a zero anchor after initialization, reconstruct it before the remaining positional rows, and optimize it with GaugeAdamW.

EVIDENCE: The adjacent fifth-to-last row failed at 73.02%, but the analogous `fc2` sequence recovered from an adjacent gauge failure by anchoring nonadjacent column 0 at 99.95%; positional row 0 therefore tests location-specific optimization sensitivity while preserving the validated initialization draw stream and exact shift symmetry.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_first = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
            full_pos_antepenultimate = self.pos_emb.weight[-3].detach().clone()
            full_pos_penultimate = self.pos_emb.weight[-2].detach().clone()
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[:-4].detach().clone()
            )
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
=======
            full_pos_first = self.pos_emb.weight[0].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
            full_pos_antepenultimate = self.pos_emb.weight[-3].detach().clone()
            full_pos_penultimate = self.pos_emb.weight[-2].detach().clone()
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[1:-4].detach().clone()
            )
            full_pos_first.sub_(full_pos_first[-1].clone())
            self.pos_emb_first.copy_(full_pos_first[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _position_weight(self) -> torch.Tensor:
        fourth_last_row = torch.cat(
=======
    def _position_weight(self) -> torch.Tensor:
        first_row = torch.cat(
            (
                self.pos_emb_first,
                self.pos_emb_first.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
            (
                self.pos_emb.weight,
                fourth_last_row,
=======
            (
                first_row,
                self.pos_emb.weight,
                fourth_last_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb_fourth_last,
=======
    gauge_parameters = [
        model.token_emb.weight,
        model.pos_emb_first,
        model.pos_emb_fourth_last,
>>>>>>> REPLACE