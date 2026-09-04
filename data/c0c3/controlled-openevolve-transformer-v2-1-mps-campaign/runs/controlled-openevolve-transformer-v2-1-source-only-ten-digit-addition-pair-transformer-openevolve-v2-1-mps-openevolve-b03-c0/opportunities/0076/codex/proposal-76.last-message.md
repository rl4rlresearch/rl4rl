MECHANISM: Trajectory-preserving positional-row shift gauge

HYPOTHESIS: Gauge-anchoring initialized positional row 1 will reduce the model from 1,582 to 1,581 parameters while retaining at least 99% accuracy after 21,000 steps.

INTENDED_EDIT: Split positional row 1 into seven learned relative coordinates plus a zero anchor after initialization, reconstruct it in its original position, and optimize it with GaugeAdamW.

EVIDENCE: Positional row 0 retained 99.98% after the adjacent fifth-to-last row failed at 73.02%; extending the identical exact symmetry to row 1 tests another position far from that failure, analogous to the successful nonadjacent recovery among `fc2` gauges.

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_first = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.pos_emb_first = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_second = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_fourth_last = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            full_pos_first = self.pos_emb.weight[0].detach().clone()
            full_pos_second = self.pos_emb.weight[1].detach().clone()
            full_pos_fourth_last = self.pos_emb.weight[-4].detach().clone()
            full_pos_antepenultimate = self.pos_emb.weight[-3].detach().clone()
            full_pos_penultimate = self.pos_emb.weight[-2].detach().clone()
            full_pos_row = self.pos_emb.weight[-1].detach().clone()
            self.pos_emb.weight = nn.Parameter(
                self.pos_emb.weight[2:-4].detach().clone()
            )
            full_pos_first.sub_(full_pos_first[-1].clone())
            self.pos_emb_first.copy_(full_pos_first[:-1])
            full_pos_second.sub_(full_pos_second[-1].clone())
            self.pos_emb_second.copy_(full_pos_second[:-1])
            full_pos_fourth_last.sub_(full_pos_fourth_last[-1].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        fourth_last_row = torch.cat(
            (
                self.pos_emb_fourth_last,
                self.pos_emb_fourth_last.new_zeros(1),
            )
        ).unsqueeze(0)
=======
        second_row = torch.cat(
            (
                self.pos_emb_second,
                self.pos_emb_second.new_zeros(1),
            )
        ).unsqueeze(0)
        fourth_last_row = torch.cat(
            (
                self.pos_emb_fourth_last,
                self.pos_emb_fourth_last.new_zeros(1),
            )
        ).unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
            (
                first_row,
                self.pos_emb.weight,
                fourth_last_row,
=======
            (
                first_row,
                second_row,
                self.pos_emb.weight,
                fourth_last_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.token_emb.weight,
        model.pos_emb_first,
        model.pos_emb_fourth_last,
=======
        model.token_emb.weight,
        model.pos_emb_first,
        model.pos_emb_second,
        model.pos_emb_fourth_last,
>>>>>>> REPLACE