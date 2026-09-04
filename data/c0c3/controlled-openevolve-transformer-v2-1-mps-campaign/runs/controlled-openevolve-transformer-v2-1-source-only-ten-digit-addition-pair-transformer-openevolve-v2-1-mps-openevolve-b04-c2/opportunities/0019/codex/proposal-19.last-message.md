MECHANISM: Incremental tail-localized positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Quotienting positional rows zero, one, and the final five rows will reduce the verified model from 1,629 to 1,628 parameters while retaining at least 99% accuracy, because the final-four-row design achieved 99.81% and the next tail row has substantially less causal influence than the failed early row-two quotient.

INTENDED_EDIT: Add the fifth-from-last positional row to the orthonormal zero-sum quotient and extend its reconstructed full-coordinate AdamW update.

EVIDENCE: Quotienting the final four rows achieved 99.81% at 1,629 parameters, while quotienting early row two achieved 98.97%; this supports the smallest additional tail-localized reduction.

<<<<<<< SEARCH
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
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final five positions, which have the shortest causal influence.
        self.compact_pos_count = 7
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-5:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-5].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
>>>>>>> REPLACE