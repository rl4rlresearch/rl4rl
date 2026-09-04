MECHANISM: Adjacent-midpoint positional-row shift gauge

HYPOTHESIS: Gauge-anchoring initialized positional row `max_seq_len // 2 + 7` will reduce the model from 1,574 to 1,573 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split the seventh row after the midpoint into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.

EVIDENCE: All seven contiguous rows from `max_seq_len // 2` through `max_seq_len // 2 + 6` succeeded with this exact gauge; the latest achieved 99.97% accuracy at 1,574 parameters, making the next adjacent row the most informative reduction.

<<<<<<< SEARCH
        self.pos_emb_middle_next_5 = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_next_6 = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb_middle_next_5 = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_next_6 = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_next_7 = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle_next_6 = self.pos_emb.weight[
                cfg.max_seq_len // 2 + 6
            ].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
=======
            full_pos_middle_next_6 = self.pos_emb.weight[
                cfg.max_seq_len // 2 + 6
            ].detach().clone()
            full_pos_middle_next_7 = self.pos_emb.weight[
                cfg.max_seq_len // 2 + 7
            ].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                        base_pos_weight[self.pos_emb_middle_index + 7 :],
=======
                        base_pos_weight[self.pos_emb_middle_index + 8 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle_next_6.sub_(full_pos_middle_next_6[-1].clone())
            self.pos_emb_middle_next_6.copy_(full_pos_middle_next_6[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
=======
            full_pos_middle_next_6.sub_(full_pos_middle_next_6[-1].clone())
            self.pos_emb_middle_next_6.copy_(full_pos_middle_next_6[:-1])
            full_pos_middle_next_7.sub_(full_pos_middle_next_7[-1].clone())
            self.pos_emb_middle_next_7.copy_(full_pos_middle_next_7[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        middle_next_6_row = torch.cat(
            (
                self.pos_emb_middle_next_6,
                self.pos_emb_middle_next_6.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
=======
        middle_next_6_row = torch.cat(
            (
                self.pos_emb_middle_next_6,
                self.pos_emb_middle_next_6.new_zeros(1),
            )
        ).unsqueeze(0)
        middle_next_7_row = torch.cat(
            (
                self.pos_emb_middle_next_7,
                self.pos_emb_middle_next_7.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
                middle_next_5_row,
                middle_next_6_row,
                self.pos_emb.weight[self.pos_emb_middle_index :],
=======
                middle_next_5_row,
                middle_next_6_row,
                middle_next_7_row,
                self.pos_emb.weight[self.pos_emb_middle_index :],
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.pos_emb_middle_next_5,
        model.pos_emb_middle_next_6,
        model.pos_emb_fourth_last,
=======
        model.pos_emb_middle_next_5,
        model.pos_emb_middle_next_6,
        model.pos_emb_middle_next_7,
        model.pos_emb_fourth_last,
>>>>>>> REPLACE