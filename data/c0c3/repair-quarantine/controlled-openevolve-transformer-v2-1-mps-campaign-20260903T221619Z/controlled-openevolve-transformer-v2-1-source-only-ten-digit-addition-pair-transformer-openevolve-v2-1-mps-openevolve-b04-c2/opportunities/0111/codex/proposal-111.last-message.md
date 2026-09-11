MECHANISM: Vocabulary-common embedding gauge quotient

HYPOTHESIS: Centering the tied token embedding across vocabulary, absorbing its removed mean into positional embeddings, and retaining the verified row-12 and head-1 value reductions will produce a 1,568-parameter model with at least 99% accuracy.

INTENDED_EDIT: Apply the qualified ninth value rotation and row-12 positional quotient, then remove eight redundant tied-embedding parameters using an orthonormal zero-sum vocabulary basis with full-coordinate AdamW updates.

EVIDENCE: The combined row-12 and head-1 rotation design achieved 99.99% accuracy at 1,576 parameters. A vocabulary-common embedding shift changes output logits only by a softmax-invariant common scalar and can be absorbed from every input by the positional embeddings, motivating an exact eight-parameter quotient without the failed deeper value or query-key constraints.

<<<<<<< SEARCH
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
=======
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-11,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 20
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:12],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[12:-9]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Remove the vocabulary-common component of the tied token embedding.
        # Its output contribution is a common logit shift, while adding its
        # mean to every positional row preserves the input representation up
        # to the common channel modes erased by LayerNorm.
        token_basis = torch.zeros(cfg.vocab_size, cfg.vocab_size - 1)
        for col in range(cfg.vocab_size - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            token_basis[: col + 1, col] = 1.0 / scale
            token_basis[col + 1, col] = -(col + 1) / scale

        # Remove LayerNorm-invariant common modes from rows 0-6, rows 8-12,
        # and the final nine rows while leaving sensitive row 7 unrestricted.
        self.compact_pos_count = 21
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_token = self.token_emb.weight.detach()
            token_mean = initialized_token.mean(dim=0)
            compact_token = token_basis.transpose(0, 1) @ initialized_token

            initialized_pos = self.pos_emb.weight.detach() + token_mean
            selected_pos = torch.cat(
                (
                    initialized_pos[:7],
                    initialized_pos[8:13],
                    initialized_pos[-9:],
                ),
                dim=0,
            )
            unrestricted_pos = torch.cat(
                (initialized_pos[7:8], initialized_pos[13:-9]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    unrestricted_pos.reshape(-1),
                )
            )

        self.lm_head = None
        self.token_emb.weight = nn.Parameter(compact_token)
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("token_basis", token_basis, persistent=False)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:11],
                unrestricted_rows[1:],
                compact_rows[11:],
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        pos_weight = torch.cat(
            (
                compact_rows[:7],
                unrestricted_rows[:1],
                compact_rows[7:12],
                unrestricted_rows[1:],
                compact_rows[12:],
            ),
            dim=0,
        )
        token_weight = self.token_basis @ self.token_emb.weight
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [
        pair
        for blk in model.blocks
=======
    gauge_params = [(model.token_emb.weight, model.token_basis)] + [
        pair
        for blk in model.blocks
>>>>>>> REPLACE

<<<<<<< SEARCH
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:11].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[11:].reshape(-1),
                    )
                )
=======
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        unrestricted_full_grad[:1].reshape(-1),
                        selected_full_grad[7:12].reshape(-1),
                        unrestricted_full_grad[1:].reshape(-1),
                        selected_full_grad[12:].reshape(-1),
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:12],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[12:-9]), dim=0
                )
=======
                selected_update = torch.cat(
                    (
                        full_update[:7],
                        full_update[8:13],
                        full_update[-9:],
                    ),
                    dim=0,
                )
                unrestricted_update = torch.cat(
                    (full_update[7:8], full_update[13:-9]), dim=0
                )
>>>>>>> REPLACE