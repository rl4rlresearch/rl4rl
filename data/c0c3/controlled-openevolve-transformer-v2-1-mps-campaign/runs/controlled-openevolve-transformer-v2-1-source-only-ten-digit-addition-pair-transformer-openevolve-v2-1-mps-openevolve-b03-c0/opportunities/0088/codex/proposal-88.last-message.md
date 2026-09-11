MECHANISM: Opposite-edge positional-row shift gauge

HYPOTHESIS: Gauge-anchoring positional row `max_seq_len // 2 - 1` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split the row immediately before the successful midpoint region into seven learned relative coordinates plus a zero anchor, reconstruct it in place, and optimize it with `GaugeAdamW`.

EVIDENCE: All eight rows from the midpoint through `+7` succeeded, reaching 99.98% at 1,573 parameters; after unsuccessful or malformed attempts beyond the upper edge, extending the identical gauge from the untested opposite edge is the most informative next reduction.

<<<<<<< SEARCH
        self.pos_emb_second = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb_second = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle_prev = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_second = self.pos_emb.weight[1].detach().clone()
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
=======
            full_pos_second = self.pos_emb.weight[1].detach().clone()
            full_pos_middle_prev = self.pos_emb.weight[
                cfg.max_seq_len // 2 - 1
            ].detach().clone()
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                        base_pos_weight[: self.pos_emb_middle_index],
                        base_pos_weight[self.pos_emb_middle_index + 8 :],
=======
                        base_pos_weight[: self.pos_emb_middle_index - 1],
                        base_pos_weight[self.pos_emb_middle_index + 8 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_second.sub_(full_pos_second[-1].clone())
            self.pos_emb_second.copy_(full_pos_second[:-1])
            full_pos_middle.sub_(full_pos_middle[-1].clone())
            self.pos_emb_middle.copy_(full_pos_middle[:-1])
=======
            full_pos_second.sub_(full_pos_second[-1].clone())
            self.pos_emb_second.copy_(full_pos_second[:-1])
            full_pos_middle_prev.sub_(full_pos_middle_prev[-1].clone())
            self.pos_emb_middle_prev.copy_(full_pos_middle_prev[:-1])
            full_pos_middle.sub_(full_pos_middle[-1].clone())
            self.pos_emb_middle.copy_(full_pos_middle[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        middle_row = torch.cat(
            (
                self.pos_emb_middle,
                self.pos_emb_middle.new_zeros(1),
            )
        ).unsqueeze(0)
=======
        middle_prev_row = torch.cat(
            (
                self.pos_emb_middle_prev,
                self.pos_emb_middle_prev.new_zeros(1),
            )
        ).unsqueeze(0)
        middle_row = torch.cat(
            (
                self.pos_emb_middle,
                self.pos_emb_middle.new_zeros(1),
            )
        ).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pos_emb.weight[: self.pos_emb_middle_index],
                middle_row,
                middle_next_row,
                middle_next_2_row,
                middle_next_3_row,
                middle_next_4_row,
                middle_next_5_row,
                middle_next_6_row,
                middle_next_7_row,
                self.pos_emb.weight[self.pos_emb_middle_index :],
=======
                self.pos_emb.weight[: self.pos_emb_middle_index - 1],
                middle_prev_row,
                middle_row,
                middle_next_row,
                middle_next_2_row,
                middle_next_3_row,
                middle_next_4_row,
                middle_next_5_row,
                middle_next_6_row,
                middle_next_7_row,
                self.pos_emb.weight[self.pos_emb_middle_index - 1 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.pos_emb_second,
        model.pos_emb_middle,
=======
        model.pos_emb_second,
        model.pos_emb_middle_prev,
        model.pos_emb_middle,
>>>>>>> REPLACE