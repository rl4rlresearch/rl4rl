MECHANISM: Adjacent-midpoint positional-row shift gauge

HYPOTHESIS: Gauge-anchoring initialized positional row `max_seq_len // 2 + 1` will reduce the model from 1,580 to 1,579 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split the row immediately after the successful midpoint row into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.

EVIDENCE: The isolated midpoint gauge achieved 99.84% at 1,580 parameters, and adjacent extensions of the same gauge succeeded across four terminal rows; testing the midpoint’s immediate neighbor is the clearest next test of whether that successful region can support another reduction despite failures at row 2 and the fifth-to-last row.

<<<<<<< SEARCH
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_next = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
=======
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
            full_pos_middle_next = self.pos_emb.weight[
                cfg.max_seq_len // 2 + 1
            ].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                        base_pos_weight[: self.pos_emb_middle_index],
                        base_pos_weight[self.pos_emb_middle_index + 1 :],
=======
                        base_pos_weight[: self.pos_emb_middle_index],
                        base_pos_weight[self.pos_emb_middle_index + 2 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle.sub_(full_pos_middle[-1].clone())
            self.pos_emb_middle.copy_(full_pos_middle[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
=======
            full_pos_middle.sub_(full_pos_middle[-1].clone())
            self.pos_emb_middle.copy_(full_pos_middle[:-1])
            full_pos_middle_next.sub_(full_pos_middle_next[-1].clone())
            self.pos_emb_middle_next.copy_(full_pos_middle_next[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        middle_row = torch.cat(
            (
                self.pos_emb_middle,
                self.pos_emb_middle.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
=======
        middle_row = torch.cat(
            (
                self.pos_emb_middle,
                self.pos_emb_middle.new_zeros(1),
            )
        ).unsqueeze(0)
        middle_next_row = torch.cat(
            (
                self.pos_emb_middle_next,
                self.pos_emb_middle_next.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pos_emb.weight[: self.pos_emb_middle_index],
                middle_row,
                self.pos_emb.weight[self.pos_emb_middle_index :],
=======
                self.pos_emb.weight[: self.pos_emb_middle_index],
                middle_row,
                middle_next_row,
                self.pos_emb.weight[self.pos_emb_middle_index :],
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.pos_emb_second,
        model.pos_emb_middle,
        model.pos_emb_fourth_last,
=======
        model.pos_emb_second,
        model.pos_emb_middle,
        model.pos_emb_middle_next,
        model.pos_emb_fourth_last,
>>>>>>> REPLACE