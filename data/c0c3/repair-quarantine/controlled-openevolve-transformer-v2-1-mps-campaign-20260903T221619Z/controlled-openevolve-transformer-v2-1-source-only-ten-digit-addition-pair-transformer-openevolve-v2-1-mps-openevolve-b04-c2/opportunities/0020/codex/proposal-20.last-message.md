MECHANISM: Incremental tail-localized positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Quotienting positional rows zero, one, and the final four rows will reproduce at least 99% accuracy with 1,629 parameters; the adjacent fifth-tail-row reduction is excluded because it fell to 96.93%.

INTENDED_EDIT: Add the fourth-from-last positional row to the orthonormal zero-sum quotient and its reconstructed full-coordinate AdamW update.

EVIDENCE: The qualified final-four-row design achieved 99.81% accuracy at 1,629 parameters, while extending the quotient to five final rows failed at 96.93%.

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final three positions, which have the shortest causal influence.
        self.compact_pos_count = 5
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-3:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-3].reshape(-1),
                )
            )
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-3:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-3].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
>>>>>>> REPLACE