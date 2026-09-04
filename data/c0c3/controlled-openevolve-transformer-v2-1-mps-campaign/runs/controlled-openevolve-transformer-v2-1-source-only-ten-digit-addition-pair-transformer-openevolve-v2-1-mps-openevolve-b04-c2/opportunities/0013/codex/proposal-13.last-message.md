MECHANISM: Dual-row positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Extending the verified 1,634-parameter design by quotienting positional row one will produce a 1,633-parameter model with at least 99% accuracy, because the single-row design reached 99.83% while quotienting all 23 rows narrowly missed at 98.16%.

INTENDED_EDIT: Parameterize positional rows zero and one in the orthonormal zero-sum basis, preserving baseline initialization and reconstructed full-coordinate AdamW dynamics.

EVIDENCE: The current single-row quotient achieved 99.83% at 1,634 parameters; the 1,612-parameter all-row quotient’s 98.16% result supports testing the smallest incremental extension rather than quotienting many additional rows simultaneously.

<<<<<<< SEARCH
        # Preserve the baseline initialization and its RNG consumption, then
        # remove only row zero's LayerNorm-invariant common-mode coordinate.
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            compact_pos = torch.cat(
                (
                    pos_basis.transpose(0, 1) @ initialized_pos[0],
                    initialized_pos[1:].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Preserve the baseline initialization and its RNG consumption, then
        # remove rows zero and one's LayerNorm-invariant common-mode coordinates.
        self.compact_pos_rows = 2
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            compact_pos = torch.cat(
                (
                    (
                        initialized_pos[: self.compact_pos_rows] @ pos_basis
                    ).reshape(-1),
                    initialized_pos[self.compact_pos_rows :].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                (self.pos_basis @ self.pos_emb.weight[: self.cfg.d_model - 1]).unsqueeze(0),
                self.pos_emb.weight[self.cfg.d_model - 1 :].view(
                    self.cfg.max_seq_len - 1, self.cfg.d_model
                ),
            ),
            dim=0,
        )
=======
        compact_size = self.compact_pos_rows * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_rows, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows,
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_rows,
                    self.cfg.d_model,
                ),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
=======
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_rows = model.compact_pos_rows
    pos_compact_size = compact_pos_rows * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        pos_basis @ pos_param.grad[: model_cfg.d_model - 1],
                        pos_param.grad[model_cfg.d_model - 1 :],
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        (
                            pos_param.grad[:pos_compact_size].view(
                                compact_pos_rows, model_cfg.d_model - 1
                            )
                            @ pos_basis.transpose(0, 1)
                        ).reshape(-1),
                        pos_param.grad[pos_compact_size:],
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                compact_update = torch.cat(
                    (
                        pos_basis.transpose(0, 1)
                        @ full_update[: model_cfg.d_model],
                        full_update[model_cfg.d_model :],
                    )
                )
=======
                full_compact_size = compact_pos_rows * model_cfg.d_model
                compact_update = torch.cat(
                    (
                        (
                            full_update[:full_compact_size].view(
                                compact_pos_rows, model_cfg.d_model
                            )
                            @ pos_basis
                        ).reshape(-1),
                        full_update[full_compact_size:],
                    )
                )
>>>>>>> REPLACE