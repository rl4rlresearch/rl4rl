MECHANISM: Bias-free LayerNorm scale absorption into unrestricted MLP input columns

HYPOTHESIS: Extending the qualified 1,562-parameter design by fixing the third bias-free terminal `ln2.weight` coordinate at one will yield 1,561 parameters while retaining at least 0.99 accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.

INTENDED_EDIT: Reproduce the qualified attention, token, positional, and LayerNorm quotients, then retain five learned `ln2` scales and reconstruct the final three as ones.

EVIDENCE: Reference Design 1 achieved 0.9993 accuracy with 1,562 parameters while fixing two terminal bias-free `ln2` scales; its 0.0093 margin motivates testing the same exact absorption on the third bias-free channel.

<<<<<<< SEARCH
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
=======
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
                elif head == 1:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the retained query
        # and output-projection biases; key shifts cancel from the softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
=======
        # Retain the qualified six-coordinate pre-attention bias quotient.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias. Remove only that single direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # The compact bias basis leaves the terminal three channels bias-free.
        # Their scales are absorbable by the corresponding unrestricted fc1
        # columns, so reconstruct those scales as fixed ones.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-3].detach().clone()
        )

        # Retain the qualified four-coordinate post-normalization bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
=======
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(3))
        )
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            ln2_weight,
            ln2_bias,
            self.ln2.eps,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of the first seven and final
        # nine positional rows.
        self.compact_pos_count = 16
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:7], initialized_pos[-9:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[7:-9].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
=======
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()
            block.mlp.compact_output_columns()

        # Remove the vocabulary-common component of the tied token embedding.
        # It contributes only a common output-logit shift, and its input
        # contribution is transferred to every positional row.
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
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[7:],
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        unrestricted_rows = self.pos_emb.weight[compact_size:].view(
            self.cfg.max_seq_len - self.compact_pos_count,
            self.cfg.d_model,
        )
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
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                full_grad = torch.cat(
                    (
                        selected_full_grad[:7].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[7:].reshape(-1),
                    )
                )
=======
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                unrestricted_full_grad = pos_param.grad[
                    pos_compact_size:
                ].view(
                    model_cfg.max_seq_len - compact_pos_count,
                    model_cfg.d_model,
                )
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
                    (full_update[:7], full_update[-9:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[7:-9].reshape(-1),
                    )
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
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        unrestricted_update.reshape(-1),
                    )
                )
>>>>>>> REPLACE