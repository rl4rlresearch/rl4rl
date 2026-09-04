MECHANISM: Twelfth positional common-mode quotient

HYPOTHESIS: Extending the verified 1,603-parameter design by quotienting the next late positional row will produce a 1,602-parameter model with at least 99% accuracy.

INTENDED_EDIT: Compact the first two and final ten positional rows, updating initialization, reconstruction, and full-coordinate AdamW projection consistently.

EVIDENCE: Reference Design 3 achieved 99.91% accuracy at 1,603 parameters after compacting the first two and final nine positional rows; the closest informative reduction extends the same successful LayerNorm-invariant quotient by one row.

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from the first two and final
        # five positional rows while preserving baseline RNG consumption.
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
=======
        # Remove LayerNorm-invariant common modes from the first two and final
        # ten positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 12
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-10:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-10].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-10:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-10].reshape(-1),
                    )
                )
>>>>>>> REPLACE