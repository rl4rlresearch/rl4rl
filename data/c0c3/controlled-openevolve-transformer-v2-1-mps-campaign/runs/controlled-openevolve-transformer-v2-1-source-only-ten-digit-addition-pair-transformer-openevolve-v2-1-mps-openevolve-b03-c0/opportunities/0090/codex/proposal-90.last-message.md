MECHANISM: Opposite-edge positional-row shift gauge with milestone-only validation

HYPOTHESIS: Gauge-anchoring positional row `max_seq_len // 2 - 1` will reduce the model from 1,573 to 1,572 parameters while retaining at least 99% accuracy, while evaluating only at schedule completion and the final step will allow the previously timed-out experiment to finish.

INTENDED_EDIT: Represent the positional row immediately before the successful midpoint region with seven learned relative coordinates and a zero anchor, optimize it with `GaugeAdamW`, and remove periodic validation passes while preserving positive-step best and last checkpoints.

EVIDENCE: Eight contiguous rows from the midpoint through `+7` retained at least 99% accuracy with the same gauge; both `-1` attempts timed out without adverse accuracy evidence, including one at a 600-step evaluation interval, motivating a much larger validation-cost reduction.

<<<<<<< SEARCH
        self.pos_emb_middle_index = cfg.max_seq_len // 2 - 2
        self.pos_emb_middle = nn.Parameter(torch.empty(cfg.d_model - 1))
=======
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
=======
            full_pos_second.sub_(full_pos_second[-1].clone())
            self.pos_emb_second.copy_(full_pos_second[:-1])
            full_pos_middle_prev.sub_(full_pos_middle_prev[-1].clone())
            self.pos_emb_middle_prev.copy_(full_pos_middle_prev[:-1])
            full_pos_middle.sub_(full_pos_middle[-1].clone())
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
                second_row,
                self.pos_emb.weight[: self.pos_emb_middle_index],
                middle_row,
=======
                second_row,
                self.pos_emb.weight[: self.pos_emb_middle_index - 1],
                middle_prev_row,
                middle_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.pos_emb.weight[self.pos_emb_middle_index :],
                fourth_last_row,
=======
                self.pos_emb.weight[self.pos_emb_middle_index - 1 :],
                fourth_last_row,
>>>>>>> REPLACE

<<<<<<< SEARCH
        model.pos_emb_second,
        model.pos_emb_middle,
=======
        model.pos_emb_second,
        model.pos_emb_middle_prev,
        model.pos_emb_middle,
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (
            (step % train_cfg.eval_interval == 0)
            or (step == schedule_steps - 1)
            or (step == train_cfg.train_steps - 1)
        ):
=======
        if (
            (step == schedule_steps - 1)
            or (step == train_cfg.train_steps - 1)
        ):
>>>>>>> REPLACE