MECHANISM: Opposite-edge positional-row shift gauge with validation-cadence thinning

HYPOTHESIS: Gauge-anchoring row `max_seq_len // 2 - 1` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy, and halving evaluation frequency will let the previously timed-out experiment complete without changing its training trajectory.

INTENDED_EDIT: Add a seven-coordinate gauge representation for the positional row immediately before the successful midpoint region, reconstruct it with a zero anchor, optimize it with `GaugeAdamW`, and change the default evaluation interval from 300 to 600 steps.

EVIDENCE: All eight contiguous rows from the midpoint through `+7` succeeded, reaching 99.98% accuracy at 1,573 parameters; the only `-1` trial timed out rather than producing adverse accuracy evidence, so repeating that reduction with fewer expensive validation passes is the most informative next test.

<<<<<<< SEARCH
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 3
        self.pos_emb_middle_prev = nn.Parameter(torch.empty(cfg.d_model - 1))
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
=======
            full_pos_middle_prev = self.pos_emb.weight[
                cfg.max_seq_len // 2 - 1
            ].detach().clone()
            full_pos_middle = self.pos_emb.weight[
                cfg.max_seq_len // 2
            ].detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
                        base_pos_weight[self.pos_emb_middle_index + 8 :],
=======
                        base_pos_weight[self.pos_emb_middle_index + 9 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_pos_middle.sub_(full_pos_middle[-1].clone())
            self.pos_emb_middle.copy_(full_pos_middle[:-1])
=======
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
=======
                self.pos_emb.weight[: self.pos_emb_middle_index],
                middle_prev_row,
                middle_row,
                middle_next_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.pos_emb_second,
        model.pos_emb_middle,
        model.pos_emb_middle_next,
=======
        model.pos_emb_second,
        model.pos_emb_middle_prev,
        model.pos_emb_middle,
        model.pos_emb_middle_next,
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=600)
>>>>>>> REPLACE